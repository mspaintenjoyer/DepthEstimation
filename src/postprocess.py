"""Post-processing utilities for disparity and depth maps."""

import cv2
import numpy as np


def filter_speckles(disparity, max_speckle_size=100, max_diff=1):
    """
    Remove small isolated regions (speckles) from disparity map.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    max_speckle_size : int
        Maximum size of speckle region to filter (pixels)
    max_diff : float
        Maximum disparity difference to consider as same region
    
    Returns:
    --------
    filtered : np.ndarray
        Filtered disparity map
    """
    filtered = disparity.copy()
    
    # Convert to 16-bit fixed-point for OpenCV
    disp_16s = (filtered * 16.0).astype(np.int16)
    
    # Filter speckles
    cv2.filterSpeckles(disp_16s, 0, max_speckle_size, int(max_diff * 16))
    
    # Convert back to float32
    filtered = disp_16s.astype(np.float32) / 16.0
    
    return filtered


def apply_bilateral_filter(disparity, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter to smooth disparity while preserving edges.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    d : int
        Diameter of pixel neighborhood
    sigma_color : float
        Filter sigma in the color space
    sigma_space : float
        Filter sigma in the coordinate space
    
    Returns:
    --------
    filtered : np.ndarray
        Filtered disparity map
    """
    # Normalize to 0-255 for better filtering
    valid_mask = disparity > 0
    if not valid_mask.any():
        return disparity
    
    disp_norm = np.zeros_like(disparity, dtype=np.uint8)
    disp_min = disparity[valid_mask].min()
    disp_max = disparity[valid_mask].max()
    
    if disp_max > disp_min:
        disp_norm[valid_mask] = ((disparity[valid_mask] - disp_min) / 
                                  (disp_max - disp_min) * 255).astype(np.uint8)
    
    # Apply bilateral filter
    filtered_norm = cv2.bilateralFilter(disp_norm, d, sigma_color, sigma_space)
    
    # Convert back to original scale
    filtered = np.zeros_like(disparity)
    filtered[valid_mask] = (filtered_norm[valid_mask].astype(np.float32) / 255.0 * 
                            (disp_max - disp_min) + disp_min)
    
    return filtered


def apply_wls_filter(disparity_left, disparity_right, left_image, 
                     lambda_=8000, sigma=1.5):
    """
    Apply Weighted Least Squares (WLS) filter for edge-aware smoothing.
    Requires both left and right disparity maps for better results.
    
    Parameters:
    -----------
    disparity_left : np.ndarray
        Disparity map from left image
    disparity_right : np.ndarray
        Disparity map from right image (for consistency check)
    left_image : np.ndarray
        Left grayscale image (used as guide image)
    lambda_ : float
        Regularization parameter (higher = smoother)
    sigma : float
        Sensitivity to edges
    
    Returns:
    --------
    filtered : np.ndarray
        Filtered disparity map
    """
    # Convert to 16-bit fixed point
    disp_left_16s = (disparity_left * 16.0).astype(np.int16)
    disp_right_16s = (disparity_right * 16.0).astype(np.int16)
    
    # Create WLS filter
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=None)
    wls_filter.setLambda(lambda_)
    wls_filter.setSigmaColor(sigma)
    
    # Apply filter
    filtered_16s = wls_filter.filter(disp_left_16s, left_image, 
                                     disparity_map_right=disp_right_16s)
    
    # Convert back to float32
    filtered = filtered_16s.astype(np.float32) / 16.0
    
    return filtered


def median_filter(disparity, kernel_size=5):
    """
    Apply median filter to remove salt-and-pepper noise.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    kernel_size : int
        Size of median filter kernel (must be odd)
    
    Returns:
    --------
    filtered : np.ndarray
        Filtered disparity map
    """
    # Preserve invalid regions
    valid_mask = disparity > 0
    
    # Convert to uint8 for median filtering
    disp_norm = np.zeros_like(disparity, dtype=np.uint8)
    if valid_mask.any():
        disp_min = disparity[valid_mask].min()
        disp_max = disparity[valid_mask].max()
        if disp_max > disp_min:
            disp_norm[valid_mask] = ((disparity[valid_mask] - disp_min) / 
                                      (disp_max - disp_min) * 255).astype(np.uint8)
    
    # Apply median filter
    filtered_norm = cv2.medianBlur(disp_norm, kernel_size)
    
    # Convert back
    filtered = np.zeros_like(disparity)
    if valid_mask.any() and disp_max > disp_min:
        filtered[valid_mask] = (filtered_norm[valid_mask].astype(np.float32) / 255.0 * 
                                (disp_max - disp_min) + disp_min)
    
    return filtered


def fill_holes(disparity, max_hole_size=10):
    """
    Fill small holes in disparity map using interpolation.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    max_hole_size : int
        Maximum hole size to fill (pixels)
    
    Returns:
    --------
    filled : np.ndarray
        Disparity map with holes filled
    """
    filled = disparity.copy()
    
    # Find invalid regions
    invalid_mask = (disparity <= 0)
    
    if not invalid_mask.any():
        return filled
    
    # Label connected components
    num_labels, labels = cv2.connectedComponents(invalid_mask.astype(np.uint8))
    
    # Fill small holes
    for label in range(1, num_labels):
        hole_mask = (labels == label)
        hole_size = hole_mask.sum()
        
        if hole_size <= max_hole_size:
            # Get surrounding valid values
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(hole_mask.astype(np.uint8), kernel)
            border = (dilated > 0) & (disparity > 0)
            
            if border.any():
                # Fill with median of surrounding values
                fill_value = np.median(disparity[border])
                filled[hole_mask] = fill_value
    
    return filled


def confidence_map(disparity_left, disparity_right, threshold=1.0):
    """
    Compute confidence map based on left-right consistency check.
    
    Parameters:
    -----------
    disparity_left : np.ndarray
        Disparity from left image
    disparity_right : np.ndarray
        Disparity from right image
    threshold : float
        Maximum allowed disparity difference
    
    Returns:
    --------
    confidence : np.ndarray
        Confidence map (0=low, 1=high)
    """
    h, w = disparity_left.shape
    confidence = np.zeros_like(disparity_left)
    
    for y in range(h):
        for x in range(w):
            d_left = disparity_left[y, x]
            if d_left <= 0:
                continue
            
            # Find corresponding point in right image
            x_right = int(x - d_left)
            if 0 <= x_right < w:
                d_right = disparity_right[y, x_right]
                # Check consistency
                if abs(d_left - d_right) < threshold:
                    confidence[y, x] = 1.0
    
    return confidence


def postprocess_disparity(disparity, left_image=None, method='bilateral', **kwargs):
    """
    Apply post-processing to disparity map.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    left_image : np.ndarray, optional
        Left image for guided filtering
    method : str
        Post-processing method: 'bilateral', 'median', 'speckle', 'fill', 'all'
    **kwargs : dict
        Additional parameters for specific methods
    
    Returns:
    --------
    processed : np.ndarray
        Post-processed disparity map
    """
    processed = disparity.copy()
    
    if method == 'speckle' or method == 'all':
        max_speckle_size = kwargs.get('max_speckle_size', 100)
        max_diff = kwargs.get('max_diff', 1)
        processed = filter_speckles(processed, max_speckle_size, max_diff)
    
    if method == 'bilateral' or method == 'all':
        d = kwargs.get('d', 9)
        sigma_color = kwargs.get('sigma_color', 75)
        sigma_space = kwargs.get('sigma_space', 75)
        processed = apply_bilateral_filter(processed, d, sigma_color, sigma_space)
    
    if method == 'median' or method == 'all':
        kernel_size = kwargs.get('kernel_size', 5)
        processed = median_filter(processed, kernel_size)
    
    if method == 'fill' or method == 'all':
        max_hole_size = kwargs.get('max_hole_size', 10)
        processed = fill_holes(processed, max_hole_size)
    
    return processed


def enhance_disparity_map(disparity, clip_percentile=1):
    """
    Enhance disparity map visualization by clipping outliers.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    clip_percentile : float
        Percentile for clipping (1-99)
    
    Returns:
    --------
    enhanced : np.ndarray
        Enhanced disparity map
    """
    valid_mask = disparity > 0
    if not valid_mask.any():
        return disparity
    
    valid_values = disparity[valid_mask]
    vmin = np.percentile(valid_values, clip_percentile)
    vmax = np.percentile(valid_values, 100 - clip_percentile)
    
    enhanced = np.clip(disparity, vmin, vmax)
    return enhanced
