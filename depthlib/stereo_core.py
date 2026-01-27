import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from depthlib.rectify import rectify_images, RectificationCache
from depthlib.postprocess import postprocess_disparity

class StereoCore:
    '''Handles common stereo operations.'''
    def __init__(self, downscale_factor=1.0, fast_mode=False) -> None:
        self.downscale_factor = downscale_factor
        self.fast_mode = fast_mode  # Skip expensive postprocessing for speed
        self.sgbm = None
        self._rect_cache = RectificationCache()  # Cache rectification maps

        # SGBM parameters with defaults
        self.sgbm_params = {
            'min_disp': 0,
            'num_disp': 128,
            'block_size': 5,
            'disp12_max_diff': 1,
            'prefilter_cap': 31,
            'uniqueness_ratio': 10,
            'speckle_window_size': 50,
            'speckle_range': 2,
            'sgbm_mode': 'sgbm_3way',  # Options: 'sgbm', 'hh', 'sgbm_3way', 'hh4'
            'focal_length': None,
            'baseline': None,
            'doffs': 0.0,
            'max_depth': None,
            'cam_matrix_L': None,
            'cam_matrix_R': None,
            'image_width': None,
            'image_height': None,
            'dist_coeff_L': None,
            'dist_coeff_R': None,
            'rotation': None,
            'translation': None,
            'hole_filling': False,
        }
        self._build_sgbm()
        self.disparity_map = None
        self.depth_map = None

    def _build_sgbm(self):
        """
        Build StereoSGBM matcher using current parameters.
        Internal method - called automatically when parameters change.
        """
        params = self.sgbm_params
        channels = 1
        P1 = 8 * channels * (params['block_size'] ** 2)
        P2 = 32 * channels * (params['block_size'] ** 2)
        
        # Select SGBM mode - HH is faster, SGBM_3WAY is highest quality
        mode_map = {
            'sgbm': cv2.STEREO_SGBM_MODE_SGBM,
            'hh': cv2.STEREO_SGBM_MODE_HH,
            'sgbm_3way': cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            'hh4': cv2.STEREO_SGBM_MODE_HH4,
        }
        mode = mode_map.get(params.get('sgbm_mode', 'sgbm_3way'), cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
        self.sgbm = cv2.StereoSGBM_create( # type: ignore
            minDisparity=params['min_disp'],
            numDisparities=params['num_disp'],
            blockSize=params['block_size'],
            P1=P1,
            P2=P2,
            disp12MaxDiff=params['disp12_max_diff'],
            preFilterCap=params['prefilter_cap'],
            uniquenessRatio=params['uniqueness_ratio'],
            speckleWindowSize=params['speckle_window_size'],
            speckleRange=params['speckle_range'],
            mode=mode,
        )

    def configure_sgbm(self, **kwargs):
        """
        Configure SGBM parameters and rebuild matcher.
        
        Parameters:
        -----------
        min_disp : int, optional
            Minimum disparity (default: 0)
        num_disp : int, optional
            Number of disparities - must be divisible by 16 (default: 128)
        block_size : int, optional
            Block size for matching (default: 5)
        disp12_max_diff : int, optional
            Maximum allowed difference in left-right disparity check (default: 1)
        prefilter_cap : int, optional
            Prefilter cap (default: 31)
        uniqueness_ratio : int, optional
            Uniqueness ratio (default: 10)
        speckle_window_size : int, optional
            Speckle window size (default: 50)
        speckle_range : int, optional
            Speckle range (default: 2)
        
        Example:
        --------
        >>> estimator.configure_sgbm(num_disp=144, block_size=7)
        >>> estimator.configure_sgbm(min_disp=16, uniqueness_ratio=15)
        """
        # Validate parameters
        valid_params = self.sgbm_params.keys()
        for key in kwargs:
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}'. Valid parameters: {list(valid_params)}")

        # Scale parameters by downscale factor if needed
        if 'num_disp' in kwargs:
            kwargs['num_disp'] = int(kwargs['num_disp'] * self.downscale_factor)
        if 'focal_length' in kwargs:
            kwargs['focal_length'] = kwargs['focal_length'] * self.downscale_factor
        if 'doffs' in kwargs:
            kwargs['doffs'] = kwargs['doffs'] * self.downscale_factor
        
        # Update parameters
        self.sgbm_params.update(kwargs)
        
        # Rebuild matcher
        self._build_sgbm()
    
    def _prepare_rectified(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rectify if calibration exists, otherwise ensure grayscale."""
        cam_matrix_L = self.sgbm_params.get('cam_matrix_L')
        cam_matrix_R = self.sgbm_params.get('cam_matrix_R')
        baseline = self.sgbm_params.get('baseline')
        img_width = self.sgbm_params.get('image_width')
        img_height = self.sgbm_params.get('image_height')
        dist_L = self.sgbm_params.get('dist_coeff_L')
        dist_R = self.sgbm_params.get('dist_coeff_R')
        rotation = self.sgbm_params.get('rotation')
        translation = self.sgbm_params.get('translation')

        if all(v is not None for v in [cam_matrix_L, cam_matrix_R, baseline, img_width, img_height]):
            left_rectified, right_rectified = rectify_images(
                left_img,
                right_img,
                cam_matrix_L,  # type: ignore
                cam_matrix_R,  # type: ignore
                baseline,  # type: ignore
                img_width,  # type: ignore
                img_height,  # type: ignore
                dist_coeff_L=dist_L,
                dist_coeff_R=dist_R,
                rotation=rotation,
                translation=translation,
                alpha=1.0,
                cache=self._rect_cache,  # Use cached rectification maps
            )
            return left_rectified, right_rectified

        # No calibration: convert to grayscale if needed
        if left_img.ndim == 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        if right_img.ndim == 3:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        return left_img, right_img

    def _process_pair(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Full pipeline for a single stereo pair (rectified input)."""

        disparity_px = self.compute_disparity(left_img, right_img)

        # crop the left invalid band
        disparity_px = disparity_px[:, self.sgbm_params['num_disp']:]

        # Fast mode: minimal postprocessing for speed
        if self.fast_mode:
            # Just a quick median filter to reduce noise
            disparity_px = cv2.medianBlur(disparity_px.astype(np.float32), 3)
        else:
            disparity_px = postprocess_disparity(
                disparity_px,
                left_image=left_img,
                max_speckle_size=int(100*self.downscale_factor),
                max_diff=1.0,
                outlier_threshold=2.5,
                fill_method='inpaint',
                apply_outlier_removal=True,
                apply_hole_filling=self.sgbm_params.get('hole_filling', False)
            )

        f_pixels = self.sgbm_params.get('focal_length', None)
        baseline_m = self.sgbm_params.get('baseline', None)
        doffs = self.sgbm_params.get('doffs', 0.0)
        min_disparity = self.sgbm_params.get('min_disp', 5.0)
        max_depth = self.sgbm_params.get('max_depth')
        depth_m = None
        if f_pixels is not None and baseline_m is not None:
            depth_m = self.disparity_to_depth(
                disparity_px, f_pixels, baseline_m, doffs,
                eps=min_disparity, max_depth=max_depth
            )

        self.disparity_map = disparity_px
        self.depth_map = depth_m
        return disparity_px, depth_m
    
    def get_sgbm_params(self) -> Dict[str, int]:
        """
        Get current SGBM parameters.
        
        Returns:
        --------
        dict : Current SGBM parameters
        """
        return self.sgbm_params.copy()
    
    def compute_disparity(self, rectified_L: np.ndarray, 
                         rectified_R: np.ndarray) -> np.ndarray:
        """
        Compute disparity map from rectified stereo images.
        
        Parameters:
        -----------
        rectified_L : np.ndarray
            Rectified left image (grayscale)
        rectified_R : np.ndarray
            Rectified right image (grayscale)
        
        Returns:
        --------
        np.ndarray : Disparity map in pixels (float32)
        """
        if self.sgbm is None:
            self._build_sgbm()
        
        disp_fixed = self.sgbm.compute(rectified_L, rectified_R) # type: ignore
        return disp_fixed.astype(np.float32) / 16.0

    def disparity_to_depth(self, disp: np.ndarray, f_pixels: float, 
                          baseline_m: float, doffs: float = 0.0, 
                          eps: float = 1e-6, max_depth: Optional[float] = None) -> np.ndarray:
        """
        Convert disparity to depth using the formula: Z = (f * B) / (d - doffs)
        
        Parameters:
        -----------
        disp : np.ndarray
            Disparity map in pixels
        f_pixels : float
            Focal length in pixels
        baseline_m : float
            Baseline distance in meters
        doffs : float
            Disparity offset (difference in principal points between cameras)
        eps : float
            Minimum valid disparity threshold (default 1e-6 is too strict)
        max_depth : float, optional
            Maximum depth value in meters. Values beyond this are clamped.
            If None, no clamping is applied.
        
        Returns:
        --------
        np.ndarray : Depth map in meters. Invalid regions are set to inf
        """
        # Adjust disparity by offset before depth calculation
        adjusted_disp = disp + doffs
        
        # Calculate depth, using inf for invalid disparities
        Z = np.full_like(disp, np.inf, dtype=np.float32)
        valid_mask = adjusted_disp > eps
        Z[valid_mask] = (f_pixels * baseline_m) / adjusted_disp[valid_mask]
        
        # Optionally clamp to maximum depth
        if max_depth is not None:
            Z[Z > max_depth] = max_depth
        
        return Z
    
    def estimate_depth(self, left_source, right_source) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate depth from stereo images.
        
        Pipeline: Raw images -> Rectification -> Disparity computation -> Depth mapping
        
        Returns:
        --------
        Tuple[np.ndarray, Optional[np.ndarray]]
            - disparity_px : Disparity map in pixels (float32)
            - depth_m : Depth map in meters (float32) or None if calibration unavailable
        """
        if left_source is None or right_source is None:
            raise ValueError("Left and right sources must be set before estimating depth.")

        self.left_rectified, self.right_rectified = self._prepare_rectified(
            left_source, right_source
        )

        return self._process_pair(self.left_rectified, self.right_rectified)