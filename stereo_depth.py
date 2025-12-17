from time import time
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
            downscale_factor=1.0,
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
        if not 0 < downscale_factor <= 1.0:
          raise ValueError(f"downscale_factor must be in (0, 1], got {downscale_factor}")
    
        if device not in ['cpu', 'cuda']:
          raise ValueError(f"device must be 'cpu' or 'cuda', got {device}")
    
        self.downscale_factor = downscale_factor
        self.left_source, self.right_source = load_stereo_pair(left_source, right_source, downscale_factor=downscale_factor)
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
            'max_depth': None,
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

        self.sgbm_params.update({
            'max_speckle_size': 100,
            'max_diff': 1.0,
            'd': 9,
            'sigma_color': 75,
            'sigma_space': 75,
            'kernel_size': 5,
            'max_hole_size': 10,
            'confidence_threshold': 0.5  
        })
    
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
        
        # Validation for new parameters
        valid_postprocess = ['max_speckle_size', 'max_diff', 'd', 'sigma_color', 
                        'sigma_space', 'kernel_size', 'max_hole_size', 'confidence_threshold']


        valid_params = self.sgbm_params.keys()
        for key in kwargs:
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}'. Valid parameters: {list(valid_params)}")
        
        # Validate num_disp is divisible by 16
        if 'num_disp' in kwargs and kwargs['num_disp'] > 280:
            raise ValueError(f"num_disp must be divisible by 16, got {kwargs['num_disp']}")
        if 'num_disp' in kwargs:
          if kwargs['num_disp'] <= 0:
            raise ValueError(f"num_disp must be positive, got {kwargs['num_disp']}")
          if kwargs['num_disp'] % 16 != 0:
            raise ValueError(f"num_disp must be divisible by 16, got {kwargs['num_disp']}")

        # Validate block_size is odd
        if 'block_size' in kwargs and kwargs['block_size'] % 2 == 0:
          raise ValueError(f"block_size must be odd, got {kwargs['block_size']}")

        # Scale disparity-dependent parameters
        if 'doffs' in kwargs:
          kwargs['doffs'] *= self.downscale_factor
        if 'min_disp' in kwargs:
          kwargs['min_disp'] = max(0.1, kwargs['min_disp'] * self.downscale_factor)
        
        self.sgbm_params.update(kwargs)    
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
                          min_disparity: float = 0.1, max_depth: Optional[float] = None) -> np.ndarray:
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
        adjusted_disp = disp - doffs
        
        with np.errstate(divide='ignore', invalid='ignore'):
        Z = (f_pixels * baseline_m) / adjusted_disp
    
        Z[(adjusted_disp <= min_disparity) | ~np.isfinite(adjusted_disp)] = np.inf
    
        if max_depth is not None:
            Z = np.minimum(Z, max_depth)
        
        return Z.astype(np.float32)
    
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
        
        # Validate image shapes match
        if self.left_source.shape[:2] != self.right_source.shape[:2]:
          raise ValueError(f"Image shape mismatch: {self.left_source.shape[:2]} vs {self.right_source.shape[:2]}")
        
        # Step 1: Rectify with proper fallback
        left_rect, right_rect = self._rectify_pair()

        # Step 2: Compute disparity
        disparity_px = self.compute_disparity(left_rect, right_rect)

        # Step 3: Left-right consistency check 
        if self.sgbm_params.get('disp12_max_diff', -1) > 0:
          
          temp_params = self.sgbm_params.copy()
          temp_params['min_disp'] = -temp_params['num_disp'] - temp_params['min_disp']

          # Compute right-to-left disparity
          temp_sgbm = cv2.StereoSGBM_create(
              minDisparity=-self.sgbm_params['num_disp'] - self.sgbm_params['min_disp'],
              numDisparities=self.sgbm_params['num_disp'],
              blockSize=self.sgbm_params['block_size']
              )
          try:
            disp_rl = temp_sgbm.compute(right_rect, left_rect).astype(np.float32) / 16.0           
            consistency_mask = np.abs(disparity_px + disp_rl) <= self.sgbm_params['disp12_max_diff'] # Mask inconsistent regions
            disparity_px[~consistency_mask] = 0  # Mark occlusions

          finally:
            del temp_sgbm

        # Step 4: Post-process with configurable parameters
        postprocess_params = {k: v for k, v in self.sgbm_params.items()
                             if k in ['max_speckle_size', 'max_diff', 'd', 'sigma_color', 'sigma_space', 'kernel_size', 'max_hole_size']}
        disparity_px = postprocess_disparity(
            disparity_px, left_image=left_rect, method='all', **postprocess_params
            )
        
        # Step 5: Compute depth if calibration available
        depth_m = self._compute_metric_depth(disparity_px) if self._has_calibration() else None
        self.disparity_map = disparity_px
        self.depth_map = depth_m
        self.left_rectified = left_rect
        self.right_rectified = right_rect
        
        return disparity_px, depth_m

    def _rectify_pair(self) -> Tuple[np.ndarray, np.ndarray]:
      """Internal rectification with proper error handling."""
      cam_matrix_L = self.sgbm_params.get('cam_matrix_L')
      baseline = self.sgbm_params.get('baseline')
      img_wh = (self.sgbm_params.get('image_width'), self.sgbm_params.get('image_height'))

      # Convert to grayscale once
      gray_left = cv2.cvtColor(self.left_source, cv2.COLOR_BGR2GRAY) if self.left_source.ndim == 3 else self.left_source
      gray_right = cv2.cvtColor(self.right_source, cv2.COLOR_BGR2GRAY) if self.right_source.ndim == 3 else self.right_source

      if all(v is not None for v in [cam_matrix_L, baseline, *img_wh]):
        return rectify_images(           
            gray_left, gray_right,
            cam_matrix_L, self.sgbm_params['cam_matrix_R'],
            baseline, *img_wh, alpha=0.0  
            )
      else:
        # Attempt uncalibrated rectification
        print("Warning: No calibration data. Attempting uncalibrated rectification...")
        
        # Find features and compute fundamental matrix
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_left, None)
        kp2, des2 = sift.detectAndCompute(gray_right, None)
        
        if len(kp1) < 50 or len(kp2) < 50:
            raise RuntimeError(f"Insufficient features : : {len(kp1)} left, {len(kp2)} right")

        # Match features
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        pts1, pts2 = [], []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
        
        if len(good_matches) < 30:
            raise RuntimeError(f"Insufficient good matches: {len(good_matches)}")
        
        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)
        
        # Compute fundamental matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)        
        h, w = gray_left.shape
        
        # Compute rectification homographies
        ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1[mask.ravel()], pts2[mask.ravel()], F, (w, h))
        
        if not ret:
            raise RuntimeError("Uncalibrated rectification failed")
        
        # Warp images
        left_rect = cv2.warpPerspective(gray_left, H1, (w, h))
        right_rect = cv2.warpPerspective(gray_right, H2, (w, h))
        
        return left_rect, right_rect           

  def _has_calibration(self) -> bool:
    """Check if metric depth computation is possible."""
    return all(self.sgbm_params.get(k) is not None 
               for k in ['focal_length', 'baseline'])

  def _compute_metric_depth(self, disparity: np.ndarray) -> np.ndarray:
    """Wrapper for depth computation with validation."""
    if not self._has_calibration():
        raise ValueError("Metric depth requires focal_length and baseline in sgbm_params")
        
    return self.disparity_to_depth(
        disparity, 
        self.sgbm_params['focal_length'],
        self.sgbm_params['baseline'],
        self.sgbm_params.get('doffs', 0.0),
        self.sgbm_params.get('min_disp', 0.1),
        self.sgbm_params.get('max_depth')
    )


    def visualize_results(self, disparity_map: np.ndarray, depth_map: Optional[np.ndarray] = None):
      """Visualize disparity and depth maps.
    
      Parameters:
      -----------
      disparity_map : np.ndarray
        Disparity map in pixels (float32)
      depth_map : np.ndarray, optional
        Depth map in meters (float32)
      """
      if disparity_map is None:
        raise ValueError("disparity_map cannot be None")
    
      visualize_disparity(disparity_map, title='Disparity Map', cmap='jet')
    
      if depth_map is not None:
        visualize_depth(depth_map, title='Depth Map', cmap='turbo_r')


    

        