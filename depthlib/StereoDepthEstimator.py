import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from depthlib.input import load_stereo_pair
from depthlib.visualizations import visualize_disparity, visualize_depth
from depthlib.rectify import rectify_images
from depthlib.postprocess import postprocess_disparity

class StereoDepthEstimator:
    '''Class for estimating depth from stereo images/videos.'''

    def __init__(
            self,
            left_source, # Path to left image/video or camera index
            right_source, # Path to right image/video or camera index
            device='cpu', # 'cpu' or 'cuda'
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
        self.left_source, self.right_source = load_stereo_pair(left_source, right_source)
        self.device = device
        
        # Store rectified images
        self.left_rectified = None
        self.right_rectified = None
        
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
            'focal_length': None,
            'baseline': None,
            'doffs': 0.0,
            'max_depth': 100.0,
            'cam_matrix_L': None,
            'cam_matrix_R': None,
            'image_width': None,
            'image_height': None,
        }
        
        # Initialize SGBM matcher
        self.sgbm = None
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
        
        self.sgbm = cv2.StereoSGBM_create(
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
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
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
        
        # Validate num_disp is divisible by 16
        if 'num_disp' in kwargs and kwargs['num_disp'] > 280:
            raise ValueError(f"num_disp must be divisible by 16, got {kwargs['num_disp']}")
        
        # Update parameters
        self.sgbm_params.update(kwargs)
        
        # Rebuild matcher
        self._build_sgbm()
    
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
        
        disp_fixed = self.sgbm.compute(rectified_L, rectified_R)
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
        adjusted_disp = disp - doffs
        
        # Calculate depth, using inf for invalid disparities
        Z = np.full_like(disp, np.inf, dtype=np.float32)
        valid_mask = adjusted_disp > eps
        Z[valid_mask] = (f_pixels * baseline_m) / adjusted_disp[valid_mask]
        
        # Optionally clamp to maximum depth
        if max_depth is not None:
            Z[Z > max_depth] = max_depth
        
        return Z
    
    def estimate_depth(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estimate depth from stereo images.
        
        Pipeline: Raw images -> Rectification -> Disparity computation -> Depth mapping
        
        Returns:
        --------
        Tuple[np.ndarray, Optional[np.ndarray]]
            - disparity_px : Disparity map in pixels (float32)
            - depth_m : Depth map in meters (float32) or None if calibration unavailable
        """
        if not hasattr(self, 'left_source') or not hasattr(self, 'right_source'):
            raise ValueError("Left and right sources must be set before estimating depth.")
        
        # Step 1: Rectify images if calibration data is available
        cam_matrix_L = self.sgbm_params.get('cam_matrix_L')
        cam_matrix_R = self.sgbm_params.get('cam_matrix_R')
        baseline = self.sgbm_params.get('baseline')
        img_width = self.sgbm_params.get('image_width')
        img_height = self.sgbm_params.get('image_height')
        
        if all(v is not None for v in [cam_matrix_L, cam_matrix_R, baseline, img_width, img_height]):
            # Rectify images
            # Type assertions safe here due to the all() check above
            self.left_rectified, self.right_rectified = rectify_images(
                self.left_source,
                self.right_source,
                cam_matrix_L,  # type: ignore
                cam_matrix_R,  # type: ignore
                baseline,  # type: ignore
                img_width,  # type: ignore
                img_height,  # type: ignore
                alpha=0.0
            )
        else:
            # If no calibration data, assume images are already rectified
            # Convert to grayscale if needed
            if self.left_source.ndim == 3:
                self.left_rectified = cv2.cvtColor(self.left_source, cv2.COLOR_BGR2GRAY)
            else:
                self.left_rectified = self.left_source
                
            if self.right_source.ndim == 3:
                self.right_rectified = cv2.cvtColor(self.right_source, cv2.COLOR_BGR2GRAY)
            else:
                self.right_rectified = self.right_source
        
        # Step 2: Compute disparity from rectified images
        disparity_px = self.compute_disparity(self.left_rectified, self.right_rectified)

        # Step 3: Post-process disparity
        disparity_px = postprocess_disparity(
            disparity_px,
            left_image=self.left_rectified,
            method='all',
            max_speckle_size=100,
            max_diff=1.0,
            d=9,
            sigma_color=75,
            sigma_space=75,
            kernel_size=5,
            max_hole_size=10
        )

        # Step 4: Compute depth if calibration data available
        f_pixels = self.sgbm_params.get('focal_length', None)
        baseline_m = self.sgbm_params.get('baseline', None)
        doffs = self.sgbm_params.get('doffs', 0.0)
        min_disparity = self.sgbm_params.get('min_disp', 0.5)
        max_depth = self.sgbm_params.get('max_depth', 100.0)
        
        depth_m = None
        if f_pixels is not None and baseline_m is not None:
            depth_m = self.disparity_to_depth(
                disparity_px, f_pixels, baseline_m, doffs,
                eps=min_disparity, max_depth=max_depth
            )
        

        self.disparity_map = disparity_px
        self.depth_map = depth_m

        return disparity_px, depth_m
    
    def visualize_results(self):
        """
        Visualize the computed disparity and depth maps.
        
        Requires that `estimate_depth` has been called.
        """
        if self.disparity_map is None:
            raise ValueError("Disparity map not computed. Call estimate_depth() first.")
        
        visualize_disparity(self.disparity_map, title='Disparity Map (Raw)', cmap='jet')

        if self.depth_map is None:
            raise ValueError("Depth map not computed. Call estimate_depth() with calibration data first.")
        
        visualize_depth(self.depth_map, title='Depth Map (Raw)', cmap='turbo_r')

    

        