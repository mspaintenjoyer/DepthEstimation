import numpy as np
from typing import Dict, Optional, Tuple
from depthlib.input import load_stereo_pair
from depthlib.visualizations import visualize_disparity, visualize_depth
from depthlib.stereo_core import StereoCore

class StereoDepthEstimator:
    '''Class for estimating depth from stereo images/videos.'''

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

        if downscale_factor <= 0 or downscale_factor > 1.0:
            raise ValueError("downscale_factor must be between 0 and 1.")
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
        Estimate depth from stereo images.
        
        Pipeline: Raw images -> Rectification -> Disparity computation -> Depth mapping
        
        Returns:
        --------
        Tuple[np.ndarray, Optional[np.ndarray]]
            - disparity_px : Disparity map in pixels (float32)
            - depth_m : Depth map in meters (float32) or None if calibration unavailable
        """
        if self.left_source is None or self.right_source is None:
            raise ValueError("Left and right sources must be provided for depth estimation.")
        disparity_px, depth_m = self.core.estimate_depth(self.left_source, self.right_source)
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