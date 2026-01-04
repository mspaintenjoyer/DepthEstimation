"""
StereoDepthEstimator: end-to-end pipeline
    rectified images -> disparity (SGBM) -> optional depth map.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import time

from .input import load_stereo_pair
from .postprocess import postprocess_disparity, compute_valid_roi
#from .rectify import rectify_images
from .visualizations import visualize_disparity, visualize_depth


class StereoDepthEstimator:
    """
    High-level wrapper for stereo depth estimation using OpenCV StereoSGBM.
    """

    def __init__(
        self,
        left_source: Union[str, np.ndarray],
        right_source: Union[str, np.ndarray],
        downscale_factor: float = 1.0,
        device: str = "cpu",
    ) -> None:
        if not (0 < downscale_factor <= 1.0):
            raise ValueError("downscale_factor must be in (0, 1].")

        self.downscale_factor = downscale_factor
        self.device = device

        if isinstance(left_source, (str, Path)):
            self.left_source, self.right_source = load_stereo_pair(
                left_source, right_source, downscale_factor=downscale_factor
            )
        else:
            # assume already-loaded images
            self.left_source = left_source
            self.right_source = right_source

        self.left_rectified: Optional[np.ndarray] = None
        self.right_rectified: Optional[np.ndarray] = None

        self.sgbm_params: Dict[str, Union[int, float, None]] = {
            "min_disp": 0,
            "num_disp": 128,
            "block_size": 5,
            "disp12_max_diff": 1,
            "prefilter_cap": 31,
            "uniqueness_ratio": 10,
            "speckle_window_size": 50,
            "speckle_range": 2,
            "focal_length": None,
            "baseline": None,
            "doffs": 0.0,
            "max_depth": None,
            "cam_matrix_L": None,
            "cam_matrix_R": None,
            "image_width": None,
            "image_height": None,
        }

        self.sgbm: Optional[cv2.StereoSGBM] = None
        self.sgbm_right: Optional[cv2.StereoSGBM] = None
        self._build_sgbm()

        self.disparity_map: Optional[np.ndarray] = None
        self.depth_map: Optional[np.ndarray] = None

    def _build_sgbm(self) -> None:
        """Create the OpenCV StereoSGBM matcher from current params."""
        params = self.sgbm_params
        channels = 1
        bs = int(params["block_size"])
        P1 = 8 * channels * (bs ** 2)
        P2 = 48 * channels * (bs ** 2)

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=int(params["min_disp"]),
            numDisparities=int(params["num_disp"]),
            blockSize=bs,
            P1=P1,
            P2=P2,
            disp12MaxDiff=int(params["disp12_max_diff"]),
            preFilterCap=int(params["prefilter_cap"]),
            uniquenessRatio=int(params["uniqueness_ratio"]),
            speckleWindowSize=int(params["speckle_window_size"]),
            speckleRange=int(params["speckle_range"]),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.sgbm_right = cv2.StereoSGBM_create(
            minDisparity=-int(params["num_disp"]),   # allow negative for R->L
            numDisparities=int(params["num_disp"]),
            blockSize=bs,
            P1=P1,
            P2=P2,
            disp12MaxDiff=int(params["disp12_max_diff"]),
            preFilterCap=int(params["prefilter_cap"]),
            uniquenessRatio=int(params["uniqueness_ratio"]),
            speckleWindowSize=int(params["speckle_window_size"]),
            speckleRange=int(params["speckle_range"]),
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def configure_sgbm(self, **kwargs) -> None:
        """
        Update SGBM parameters.

        Examples
        --------
        estimator.configure_sgbm(num_disp=256, block_size=7, uniqueness_ratio=5)
        """
        valid_params = self.sgbm_params.keys()
        for key in kwargs:
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}'. Valid: {list(valid_params)}")

        if "num_disp" in kwargs:
            nd = int(kwargs["num_disp"] * self.downscale_factor)
            nd = max(16, (nd // 16) * 16)  # round down to multiple of 16
            kwargs["num_disp"] = nd
        if "focal_length" in kwargs:
            kwargs["focal_length"] = kwargs["focal_length"] * self.downscale_factor
        if "doffs" in kwargs:
            kwargs["doffs"] = kwargs["doffs"] * self.downscale_factor

        self.sgbm_params.update(kwargs)
        self._build_sgbm()

    def compute_disparity(self, rectified_L: np.ndarray, rectified_R: np.ndarray) -> np.ndarray:
        """Run SGBM and return disparity in pixels (float32)."""
        if self.sgbm is None:
            self._build_sgbm()
        disp_fixed = self.sgbm.compute(rectified_L, rectified_R)
        return disp_fixed.astype(np.float32) / 16.0

    def disparity_to_depth(
        self,
        disp: np.ndarray,
        f_pixels: float,
        baseline_m: float,
        doffs: float = 0.0,
        eps: float = 1e-6,
        max_depth: Optional[float] = None,
    ) -> np.ndarray:
        """Convert disparity to depth Z = f * B / (d + doffs)."""
        adjusted_disp = disp + doffs
        Z = np.full_like(disp, np.inf, dtype=np.float32)
        valid = adjusted_disp > eps
        Z[valid] = (f_pixels * baseline_m) / adjusted_disp[valid]
        if max_depth is not None:
            Z[Z > max_depth] = max_depth
        return Z

    def estimate_depth(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Full pipeline: rectification (if calib), SGBM, post-process, depth (if calib).
        """
        # Always convert to grayscale (Middlebury pairs are already rectified)
        L = self.left_source
        R = self.right_source

        if L.ndim == 3:
            L = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        if R.ndim == 3:
            R = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

        # Optional: enforce a known working resolution (from calib) if provided
        img_width = self.sgbm_params.get("image_width")
        img_height = self.sgbm_params.get("image_height")
        if img_width is not None and img_height is not None:
            L = cv2.resize(L, (int(img_width), int(img_height)), interpolation=cv2.INTER_AREA)
            R = cv2.resize(R, (int(img_width), int(img_height)), interpolation=cv2.INTER_AREA)

        self.left_rectified = L
        self.right_rectified = R


        disparity_L = self.compute_disparity(self.left_rectified, self.right_rectified)

        # Right disparity (R->L). Note: uses sgbm_right directly because minDisparity differs.
        dispR_fixed = self.sgbm_right.compute(self.right_rectified, self.left_rectified)  # type: ignore
        disparity_R = dispR_fixed.astype(np.float32) / 16.0

        disparity_px = postprocess_disparity(
        disparity_L,
        disparity_R=disparity_R,
        apply_lr_consistency=True,
        lr_thresh=1.0,
        apply_speckle_filter=True,
        apply_fill_from_right=False,
        invalidate_value=-1.0,
        )

        # --- ROI crop to remove invalid left band ---
        invalid_value = -1.0
        x0, x1 = compute_valid_roi(disparity_px, invalid_value=invalid_value, min_valid_frac=0.60)
        disparity_px = disparity_px[:, x0:x1]
        # ------------------------------------------

        # Optional depth
        f_pixels = self.sgbm_params.get("focal_length", None)
        baseline_m = self.sgbm_params.get("baseline", None)
        doffs = self.sgbm_params.get("doffs", 0.0)
        min_disparity = self.sgbm_params.get("min_disp", 5.0)
        max_depth = self.sgbm_params.get("max_depth")

        depth_m: Optional[np.ndarray] = None
        if depth_m is not None:
            depth_m = depth_m[:, x0:x1]

        if f_pixels is not None and baseline_m is not None:
            depth_m = self.disparity_to_depth(
                disparity_px,
                float(f_pixels),
                float(baseline_m),
                doffs=float(doffs),
                eps=float(min_disparity),
                max_depth=max_depth,
            )

        self.disparity_map = disparity_px
        self.depth_map = depth_m

        return disparity_px, depth_m

    def visualize_results(self) -> None:
        """Show disparity and depth maps (if computed)."""
        if self.disparity_map is None:
            raise ValueError("Disparity map not computed. Call estimate_depth() first.")

        visualize_disparity(self.disparity_map, title="Disparity Map (Raw)", cmap="jet")

        if self.depth_map is None:
            raise ValueError(
                "Depth map not computed. Call estimate_depth() with calibration data first."
            )
        visualize_depth(self.depth_map, title="Depth Map (Raw)", cmap="turbo_r")
