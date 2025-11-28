"""Post-processing utilities for disparity and depth maps."""

import cv2
import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, fastmath=True)
def _bilateral_filter_numba(disparity, output, d, sigma_color, sigma_space):
    """
    Numba-accelerated bilateral filter implementation.
    Applies edge-preserving smoothing in parallel.
    """
    h, w = disparity.shape
    radius = d // 2
    
    for y in prange(h):  # Parallel loop over rows
        for x in range(w):
            center_val = disparity[y, x]
            
            if center_val <= 0:
                output[y, x] = 0.0
                continue
            
            sum_val = 0.0
            weight_sum = 0.0
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny = y + dy
                    nx = x + dx
                    
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor_val = disparity[ny, nx]
                        
                        if neighbor_val > 0:
                            # Spatial distance
                            space_dist = np.sqrt(dx * dx + dy * dy)
                            # Color distance
                            color_dist = abs(center_val - neighbor_val)
                            
                            # Gaussian weights
                            space_weight = np.exp(-space_dist * space_dist / (2 * sigma_space * sigma_space))
                            color_weight = np.exp(-color_dist * color_dist / (2 * sigma_color * sigma_color))
                            
                            weight = space_weight * color_weight
                            sum_val += neighbor_val * weight
                            weight_sum += weight
            
            if weight_sum > 0:
                output[y, x] = sum_val / weight_sum
            else:
                output[y, x] = center_val
    
    return output


@jit(nopython=True, parallel=True)
def _median_filter_3x3_numba(disparity, output):
    """
    Fast 3x3 median filter using Numba parallelization.
    """
    h, w = disparity.shape
    
    for y in prange(1, h - 1):  # Parallel loop, skip borders
        for x in range(1, w - 1):
            # Collect 3x3 neighborhood
            values = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    val = disparity[y + dy, x + dx]
                    if val > 0:  # Only consider valid disparities
                        values.append(val)
            
            if len(values) > 0:
                # Simple median calculation
                values_sorted = sorted(values)
                n = len(values_sorted)
                if n % 2 == 1:
                    output[y, x] = values_sorted[n // 2]
                else:
                    output[y, x] = (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2.0
            else:
                output[y, x] = disparity[y, x]
    
    return output


@jit(nopython=True, parallel=True)
def _fill_small_holes_numba(disparity, output, max_radius=2):
    """
    Fill small holes in disparity map using neighborhood interpolation.
    Parallelized using Numba.
    """
    h, w = disparity.shape
    
    for y in prange(h):  # Parallel loop over rows
        for x in range(w):
            if disparity[y, x] > 0:
                output[y, x] = disparity[y, x]
            else:
                # Hole detected, try to fill with neighborhood average
                sum_val = 0.0
                count = 0
                
                for dy in range(-max_radius, max_radius + 1):
                    for dx in range(-max_radius, max_radius + 1):
                        ny = y + dy
                        nx = x + dx
                        
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbor_val = disparity[ny, nx]
                            if neighbor_val > 0:
                                sum_val += neighbor_val
                                count += 1
                
                if count > 0:
                    output[y, x] = sum_val / count
                else:
                    output[y, x] = 0.0
    
    return output


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


def apply_bilateral_filter(disparity, d=9, sigma_color=75, sigma_space=75, use_numba=True):
    """
    Apply bilateral filter to smooth disparity while preserving edges.
    Uses Numba-accelerated implementation for better performance.
    
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
    use_numba : bool
        Use Numba-accelerated version (default: True)
    
    Returns:
    --------
    filtered : np.ndarray
        Filtered disparity map
    """
    if use_numba:
        # Use parallelized Numba implementation
        output = np.zeros_like(disparity, dtype=np.float32)
        _bilateral_filter_numba(disparity.astype(np.float32), output, d, 
                                float(sigma_color), float(sigma_space))
        return output
    else:
        # Fallback to OpenCV implementation
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


def median_filter(disparity, kernel_size=5, use_numba=True):
    """
    Apply median filter to remove salt-and-pepper noise.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    kernel_size : int
        Size of median filter kernel (must be odd)
    use_numba : bool
        Use Numba-accelerated 3x3 version when kernel_size=3 (default: True)
    
    Returns:
    --------
    filtered : np.ndarray
        Filtered disparity map
    """
    # Use fast Numba version for 3x3 kernel
    if use_numba and kernel_size == 3:
        output = np.zeros_like(disparity, dtype=np.float32)
        _median_filter_3x3_numba(disparity.astype(np.float32), output)
        return output
    
    # Fallback to OpenCV for other kernel sizes
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
        else:
            return disparity
    
        # Apply median filter
        filtered_norm = cv2.medianBlur(disp_norm, kernel_size)
        
        # Convert back
        filtered = np.zeros_like(disparity)
        filtered[valid_mask] = (filtered_norm[valid_mask].astype(np.float32) / 255.0 * 
                                    (disp_max - disp_min) + disp_min)
        
        return filtered
    
    return disparity


def fill_holes(disparity, max_hole_size=10, use_numba=True):
    """
    Fill small holes in disparity map using interpolation.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    max_hole_size : int
        Maximum hole size to fill (pixels)
    use_numba : bool
        Use fast Numba-parallelized version (default: True)
    
    Returns:
    --------
    filled : np.ndarray
        Disparity map with holes filled
    """
    # Use fast Numba version for simple hole filling
    if use_numba:
        output = np.zeros_like(disparity, dtype=np.float32)
        # Use max_radius = 2 (5x5 neighborhood) for fast filling
        _fill_small_holes_numba(disparity.astype(np.float32), output, max_radius=2)
        return output
    
    # Original OpenCV-based approach (slower but more sophisticated)
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


@jit(nopython=True, parallel=True, fastmath=True)
def confidence_map(disparity_left, disparity_right, threshold=1.0):
    """
    Compute confidence map based on left-right consistency check.
    Parallelized using Numba for faster processing.
    
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
    
    for y in prange(h):  # Parallel loop over rows
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


def postprocess_disparity(disparity, left_image=None, method='all', **kwargs):
    """
    Apply post-processing to disparity map with parallel execution.
    
    Parallelization Strategy:
    - PARALLEL: speckle, bilateral, and median filters (independent - run simultaneously using Numba)
    - SEQUENTIAL: fill_holes (dependent - runs on combined result)
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    left_image : np.ndarray, optional
        Left image for guided filtering (not used currently)
    method : str
        Post-processing method: 'bilateral', 'median', 'speckle', 'fill', 'all'
        Default: 'all' (runs all filters with parallelization)
    **kwargs : dict
        Additional parameters for specific methods:
        - max_speckle_size : int (default: 100)
        - max_diff : float (default: 1)
        - d : int (default: 9)
        - sigma_color : float (default: 75)
        - sigma_space : float (default: 75)
        - kernel_size : int (default: 5)
        - max_hole_size : int (default: 10)
        - combine_method : str (default: 'weighted_average')
          Options: 'weighted_average', 'median', 'average'
        - w_speckle : float (default: 0.2)
        - w_bilateral : float (default: 0.5)
        - w_median : float (default: 0.3)
    
    Returns:
    --------
    processed : np.ndarray
        Post-processed disparity map
    """
    
    if method == 'all':
        # === STEP 1: Run independent filters ===
        
        print("Running independent filters with parallelization...")
        
        # Filter 1: Remove small isolated regions
        print("  [1/4] Speckle filtering...")
        result_speckle = filter_speckles(
            disparity.copy(),
            kwargs.get('max_speckle_size', 100),
            kwargs.get('max_diff', 1)
        )
        
        # Filter 2: Edge-preserving smoothing (Numba-parallelized)
        print("  [2/4] Bilateral filtering...")
        result_bilateral = apply_bilateral_filter(
            disparity.copy(),
            kwargs.get('d', 9),
            kwargs.get('sigma_color', 75),
            kwargs.get('sigma_space', 75),
            use_numba=True
        )
        
        # Filter 3: Noise removal (Numba-parallelized)
        print("  [3/4] Median filtering...")
        result_median = median_filter(
            disparity.copy(),
            kwargs.get('kernel_size', 5),
            use_numba=True
        )
        
        print("✓ All independent filters completed!")
        
        # === STEP 2: Combine the parallel results ===
        combine_method = kwargs.get('combine_method', 'weighted_average')
        
        if combine_method == 'weighted_average':
            # Weighted average: prioritize bilateral (best edge preservation)
            w_speckle = kwargs.get('w_speckle', 0.2)
            w_bilateral = kwargs.get('w_bilateral', 0.5)
            w_median = kwargs.get('w_median', 0.3)
            
            processed = (w_speckle * result_speckle + 
                        w_bilateral * result_bilateral + 
                        w_median * result_median)
        
        elif combine_method == 'median':
            # Median of three results (most robust to outliers)
            stacked = np.stack([result_speckle, result_bilateral, result_median])
            processed = np.median(stacked, axis=0).astype(np.float32)
        
        else:  # 'average'
            # Simple average
            processed = (result_speckle + result_bilateral + result_median) / 3.0
        
        # === STEP 3: Apply hole filling (SEQUENTIAL - depends on filtered results) ===
        # This must run AFTER filtering to work effectively
        print("  [4/4] Hole filling (Numba-parallelized)...")
        max_hole_size = kwargs.get('max_hole_size', 10)
        processed = fill_holes(processed, max_hole_size, use_numba=True)
        print("✓ Post-processing complete!")
        
        return processed
    
    else:
        # Single filter mode (no parallelization needed)
        processed = disparity.copy()
        
        if method == 'speckle':
            max_speckle_size = kwargs.get('max_speckle_size', 100)
            max_diff = kwargs.get('max_diff', 1)
            processed = filter_speckles(processed, max_speckle_size, max_diff)
        
        elif method == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            processed = apply_bilateral_filter(processed, d, sigma_color, sigma_space)
        
        elif method == 'median':
            kernel_size = kwargs.get('kernel_size', 5)
            processed = median_filter(processed, kernel_size)
        
        elif method == 'fill':
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
