"""Post-processing utilities for disparity and depth maps."""
from scipy.ndimage import binary_dilation, median_filter
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

def detect_outliers(disparity, threshold=3.0, kernel_size=5):
    """
    Detect outliers in disparity map using local statistics.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    threshold : float
        Number of standard deviations for outlier detection
    kernel_size : int
        Size of local neighborhood for statistics
        
    Returns:
    --------
    mask : np.ndarray (bool)
        Boolean mask where True indicates outliers
    """
    # Create a mask for valid disparities
    valid_mask = disparity > 0
    
    # Compute local mean and std using box filter
    mean = cv2.boxFilter(disparity, -1, (kernel_size, kernel_size))
    
    # Compute local standard deviation
    disparity_sq = disparity ** 2
    mean_sq = cv2.boxFilter(disparity_sq, -1, (kernel_size, kernel_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    
    # Detect outliers
    diff = np.abs(disparity - mean)
    outlier_mask = (diff > threshold * std) & valid_mask
    
    return outlier_mask

def fill_holes(disparity, mask=None, method='inpaint', kernel_size=5):
    """
    Fill holes and invalid regions in disparity map.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    mask : np.ndarray (bool), optional
        Boolean mask indicating holes to fill (True = hole)
        If None, fills all zero/invalid values
    method : str
        Filling method: 'inpaint' or 'nearest'
    kernel_size : int
        Kernel size for morphological operations
        
    Returns:
    --------
    filled : np.ndarray
        Disparity map with filled holes
    """
    filled = disparity.copy()
    
    # Create hole mask if not provided
    if mask is None:
        mask = (disparity <= 0)
    
    # Convert mask to uint8 for OpenCV
    hole_mask = mask.astype(np.uint8) * 255
    
    if method == 'inpaint':
        # Use Telea or NS inpainting algorithm
        filled = cv2.inpaint(filled.astype(np.float32), hole_mask, 
                            kernel_size, cv2.INPAINT_TELEA)
    elif method == 'nearest':
        # Dilate valid regions to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (kernel_size, kernel_size))
        # Create distance transform from valid pixels
        dist = cv2.distanceTransform((~mask).astype(np.uint8), 
                                     cv2.DIST_L2, 5)
        # Fill with nearest valid neighbor
        for _ in range(kernel_size):
            dilated = cv2.dilate(filled, kernel)
            filled = np.where(mask, dilated, filled)
    
    return filled

def fill_left_band(disparity, invalid_value=-1.0, max_search=200):
    """
    Detect a contiguous invalid band on the left and fill it from the right.
    Only processes rows and columns that actually contain the band.
    """
    disp = disparity.copy().astype(np.float32)
    H, W = disp.shape

    # 1) Detect band width per row (up to max_search)
    band_widths = np.zeros(H, dtype=np.int32)
    any_band = False

    for y in range(H):
        row = disp[y, :max_search]
        # find first valid index in this row
        val_idx = np.where(row != invalid_value)[0]
        if val_idx.size == 0:
            continue  # this row is all invalid on the left, skip for now
        w = val_idx[0]
        if w > 0:
            band_widths[y] = w
            any_band = True

    if not any_band:
        return disp  # nothing to do

    # 2) For rows where band exists, fill only that band
    for y in range(H):
        w = band_widths[y]
        if w == 0:
            continue
        # we know column w is the first valid; its value is the source
        src_val = disp[y, w]
        disp[y, :w] = src_val

    return disp


def postprocess_disparity(disparity, **kwargs):
    """
    Apply a series of post-processing steps to refine the disparity map.

    Additional kwargs:
    - invalidate_value: float (default: 0.0)
        Value considered invalid for band-filling / hole-filling.
    - apply_fill_from_right: bool (default: False)
        If True, fill invalid left bands by propagating disparities from the right.
    """
    invalid_value = kwargs.get('invalidate_value', -1.0)
    band_width = kwargs.get('band_width', 40) #ye naya hai
    #orig_invalid = (disparity <= invalid_value)

    # Step 1: Quick speckle removal
    #result = filter_speckles(
    #    disparity.copy(),
    #    kwargs.get('max_speckle_size', 50),
    #    kwargs.get('max_diff', 1)
    #)
    result = disparity.copy()

    # Step 1.5: optional left-band fix via horizontal fill
    if kwargs.get('apply_fill_from_right', False):
        result = fill_left_band(result, invalid_value=invalid_value)
        
    #if kwargs.get('apply_targeted_median', False):        # <-- add
    #    band = binary_dilation(orig_invalid, iterations=kwargs.get('band_dilate', 1))
    #    smoothed = median_filter(result, size=kwargs.get('median_size', 3))
    #    result[band] = smoothed[band]

    # Step 2: Outlier detection and masking
    #if kwargs.get('apply_outlier_removal', True):
     #   outlier_mask = detect_outliers(
     #       result,
      #      threshold=kwargs.get('outlier_threshold', 3.0),
      #      kernel_size=kwargs.get('outlier_kernel', 5)
      #  )
        # Set outliers to invalid_value
      #  result[outlier_mask] = invalid_value

    # Step 3: Hole filling (inpaint/nearest) treating <= invalid_value as holes
    #if kwargs.get('apply_hole_filling', True):
     #   hole_mask = (result <= invalid_value)
     #   result = fill_holes(
      #      result,
      #      mask=hole_mask,
      #      method=kwargs.get('fill_method', 'inpaint'),
      #      kernel_size=kwargs.get('fill_kernel', 5)
       # )

    # Step 4: Fast 3x3 median filter
    #if kwargs.get('apply_median', False):
    #    result = cv2.medianBlur(result.astype(np.float32), 3)
    return result

#def fill_from_right(disparity, invalid_value=-1.0, max_col=None):
#    """
#    Fill invalid disparities by propagating values horizontally from right to left.
#    Optionally restrict to columns [0, max_col) to speed up.
#    """
 #   disp = disparity.copy().astype(np.float32)
 #   H, W = disp.shape

 #   if max_col is None or max_col > W:
#        max_col = W

#    for y in range(H):
 #       last_valid = None
        # only scan up to max_col from right side
#        for x in range(max_col - 1, -1, -1):
#            if disp[y, x] != invalid_value:
#                last_valid = disp[y, x]
#            elif last_valid is not None:
#                disp[y, x] = last_valid

#    return disp



