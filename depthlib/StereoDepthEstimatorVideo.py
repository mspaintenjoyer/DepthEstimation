from depthlib.StereoDepthEstimator import StereoDepthEstimator
from depthlib.input import stereo_stream
from depthlib.visualizations import visualize_stereo_live
import cv2
import time

class StereoDepthEstimatorVideo:
    '''class for estimating depth from stereo video streams'''

    def __init__(
        self,
        left_source=None, # Path to left video
        right_source=None, # Path to right video
        downscale_factor=1.0,
        device='cpu', # 'cpu' or 'cuda'
        visualize_live=False,
        saving_path=None, # Path to save output video
    ) -> None:
        '''Initialize the StereoDepthEstimatorVideo with video sources and parameters.'''
        self.left_source = left_source
        self.right_source = right_source
        self.downscale_factor = downscale_factor
        self.device = device
        self.visualize_live = visualize_live
        self.saving_path = saving_path
        
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

    def estimate_depth(self):
        '''Estimate depth from the stereo video streams.'''
        if self.left_source is None or self.right_source is None:
            raise ValueError("Both left_source and right_source must be provided for video depth estimation.")
        estimator = StereoDepthEstimator(
            left_source=None,
            right_source=None,
            downscale_factor=self.downscale_factor,
            device=self.device,
        )
        estimator.configure_sgbm(**self.sgbm_params)

        #Allow window resizing
        cv2.namedWindow("Depth (live)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth (live)", 960, 540)
        for idx, (left_frame, right_frame) in enumerate(stereo_stream(self.left_source, self.right_source, downscale_factor=self.downscale_factor)):
            t0 = time.time()
            disparity_px, depth_m = estimator.estimate_depth_frame(left_frame, right_frame)
            fps = 1.0 / max(time.time() - t0, 1e-6)

            if self.visualize_live:
                visualize_stereo_live(depth_m, fps)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
                cv2.destroyAllWindows()
                break