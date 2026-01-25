from depthlib.StereoDepthEstimator import StereoDepthEstimator
from depthlib.input import stereo_stream
from depthlib.visualizations import visualize_depth_live_gray
from depthlib.stereo_core import StereoCore
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
        
        self.core = StereoCore(downscale_factor=downscale_factor, device=device)

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
        self.core.configure_sgbm(**kwargs)

    def estimate_depth(self):
        '''Estimate depth from the stereo video streams.'''
        if self.left_source is None or self.right_source is None:
            raise ValueError("Both left_source and right_source must be provided for video depth estimation.")

        self.core.configure_sgbm(**self.core.get_sgbm_params())

        #Allow window resizing
        cv2.namedWindow("Depth (live)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth (live)", 960, 540)
        for left_frame, right_frame in stereo_stream(self.left_source, self.right_source, downscale_factor=self.downscale_factor):
            t0 = time.time()
            disparity_px, depth_m = self.core.estimate_depth(left_frame, right_frame)
            fps = 1.0 / max(time.time() - t0, 1e-6)

            if self.visualize_live:
                visualize_depth_live_gray(depth_m, fps)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
                cv2.destroyAllWindows()
                break