import numpy as np
from typing import Dict, Optional, Tuple
from depthlib.input import load_stereo_pair
from depthlib.visualizations import visualize_disparity, visualize_depth
from depthlib.stereo_core import StereoCore

class StereoDepthEstimator:
    """
    High-level wrapper for stereo depth estimation using OpenCV StereoSGBM.
    """

    def __init__(
        self,
        left_source=None, # Path to left image
        right_source=None, # Path to right image
        downscale_factor=1.0,
    ):
        """
        Initialize the StereoDepthEstimator.
        
        Parameters:
        -----------
        left_source : str or int
            Path to left image/video or camera index
        right_source : str or int
            Path to right image/video or camera index
        device : str
            Device to use ('cpu' or 'cuda')
        calibration_data : dict, optional
            Calibration data dictionary
        calibration_file : str, optional
            Path to calibration file
        """

        self.downscale_factor = downscale_factor

        self.core = StereoCore(downscale_factor=downscale_factor)

        self.left_source = None
        self.right_source = None
        if left_source is not None and right_source is not None:
            self.left_source, self.right_source = load_stereo_pair(left_source, right_source, downscale_factor=downscale_factor)
        
        # Initialize SGBM matcher
        self.sgbm = None
        self.disparity_map = None
        self.depth_map = None

    def configure_sgbm(self, **kwargs) -> None:
        """
        Update SGBM parameters.

        Examples
        --------
        estimator.configure_sgbm(num_disp=256, block_size=7, uniqueness_ratio=5)
        """
        # Validate parameters
        self.core.configure_sgbm(**kwargs)
    
    def get_sgbm_params(self) -> Dict[str, int]:
        """
        Get current SGBM parameters.
        
        Returns:
        --------
        dict : Current SGBM parameters
        """
        return self.core.get_sgbm_params()
    
    def estimate_depth(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Full pipeline: rectification (if calib), SGBM, post-process, depth (if calib).
        """
            disparity_px,
            apply_fill_from_right=True,
            invalidate_value=-1.0,
        # num_disp = self.sgbm_params.get('num_disp', 128)
        # min_disp = self.sgbm_params.get('min_disp', 0)
        # crop_width = num_disp + min_disp
        
        # # Crop disparity map
        # disparity_px = disparity_px[:, crop_width:]
        
        # # Also crop the rectified images for visualization consistency
        # self.left_rectified = self.left_rectified[:, crop_width:]
        # self.right_rectified = self.right_rectified[:, crop_width:]

        # Step 3: Post-process disparity
        disparity_px = postprocess_disparity(
            disparity_px,
            left_image=self.left_rectified,
            max_speckle_size=int(100*self.downscale_factor),
            max_diff=1.0,
            outlier_threshold=2.5,
            fill_method='inpaint',
            apply_outlier_removal=False,
            apply_hole_filling=False
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
        # Step 4: Compute depth if calibration data available
        f_pixels = self.sgbm_params.get('focal_length', None)
        baseline_m = self.sgbm_params.get('baseline', None)
        doffs = self.sgbm_params.get('doffs', 0.0)
        min_disparity = self.sgbm_params.get('min_disp', 5.0)
        max_depth = self.sgbm_params.get('max_depth')
        
        depth_m = None
        if f_pixels is not None and baseline_m is not None:
            depth_m = self.disparity_to_depth(
                disparity_px,
                float(f_pixels),
                float(baseline_m),
                doffs=float(doffs),
                eps=float(min_disparity),
                max_depth=max_depth,
            )


        if self.left_source is None or self.right_source is None:
            raise ValueError("Left and right sources must be provided for depth estimation.")
        disparity_px, depth_m = self.core.estimate_depth(self.left_source, self.right_source)
        self.disparity_map = disparity_px
        self.depth_map = depth_m
        return disparity_px, depth_m

    def visualize_results(self) -> None:
        """Show disparity and depth maps (if computed)."""
        if self.disparity_map is None:
            raise ValueError("Disparity map not computed. Call estimate_depth() first.")

        if self.depth_map is None:
            raise ValueError("Depth map not computed. Call estimate_depth() with calibration data first.")
        
        visualize_depth(self.depth_map, title='Depth Map (Raw)', cmap='turbo_r')
