"""
Post-processing utilities for disparity maps.
"""

from typing import Optional

import cv2
import numpy as np


def filter_speckles(
    disparity: np.ndarray, max_speckle_size: int = 100, max_diff: float = 1.0
) -> np.ndarray:
    """
    Remove small isolated regions (speckles) from disparity map.

    Parameters
    ----------
    disparity : np.ndarray
        Input disparity map (float32).
    max_speckle_size : int
        Maximum size of speckle region to filter (pixels).
    max_diff : float
        Maximum disparity difference to consider as same region.

    Returns
    -------
    filtered : np.ndarray
        Filtered disparity map.
    """
    filtered = disparity.copy()
    disp_16s = (filtered * 16.0).astype(np.int16)
    cv2.filterSpeckles(disp_16s, 0, max_speckle_size, int(max_diff * 16))
    return disp_16s.astype(np.float32) / 16.0


def detect_outliers(
    disparity: np.ndarray, threshold: float = 3.0, kernel_size: int = 5
) -> np.ndarray:
    """
    Detect outliers using local mean/std.

    Returns
    -------
    outlier_mask : np.ndarray (bool)
        True where disparity is considered an outlier.
    """
    valid_mask = disparity > 0
    mean = cv2.boxFilter(disparity, -1, (kernel_size, kernel_size))
    disparity_sq = disparity ** 2
    mean_sq = cv2.boxFilter(disparity_sq, -1, (kernel_size, kernel_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    diff = np.abs(disparity - mean)
    return (diff > threshold * std) & valid_mask


def fill_holes(
    disparity: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "inpaint",
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Fill holes / invalid regions in a disparity map.

    Parameters
    ----------
    disparity : np.ndarray
        Input disparity map.
    mask : np.ndarray (bool), optional
        True for holes to fill. If None, disparity <= 0 is treated as holes.
    method : {"inpaint", "nearest"}
        Filling strategy.
    kernel_size : int
        Kernel size for morphological operations / inpainting radius.

    Returns
    -------
    filled : np.ndarray
        Filled disparity map.
    """
    filled = disparity.copy()

    if mask is None:
        mask = disparity <= 0

    hole_mask = mask.astype(np.uint8) * 255

    if method == "inpaint":
        filled = cv2.inpaint(filled.astype(np.float32), hole_mask, kernel_size, cv2.INPAINT_TELEA)
    elif method == "nearest":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        for _ in range(kernel_size):
            dilated = cv2.dilate(filled, kernel)
            filled = np.where(mask, dilated, filled)

    return filled


def fill_left_band(
    disparity: np.ndarray, invalid_value: float = -1.0, max_search: int = 200
) -> np.ndarray:
    """
    Detect a contiguous invalid band at the left border and fill it from the right.

    Only rows that actually contain such a band are modified.

    Parameters
    ----------
    disparity : np.ndarray
        Input disparity map (float32).
    invalid_value : float
        Value used to encode invalid disparities (e.g., -1.0).
    max_search : int
        Maximum number of leftmost columns to inspect for the band.

    Returns
    -------
    disp : np.ndarray
        Disparity map with left band filled.
    """
        
    # Step 1: Quick speckle removal
    result = filter_speckles(
        disparity.copy(),
        kwargs.get('max_speckle_size', 50),
        kwargs.get('max_diff', 1)
    )
    
    # Step 2: Outlier detection and masking
    if kwargs.get('apply_outlier_removal', True):
        outlier_mask = detect_outliers(
            result,
            threshold=kwargs.get('outlier_threshold', 3.0),
            kernel_size=kwargs.get('outlier_kernel', 5)
        )
        # Set outliers to zero (invalid)
        result[outlier_mask] = 0
    
    # Step 3: Hole filling
    if kwargs.get('apply_hole_filling', True):
        result = fill_holes(
            result,
            method=kwargs.get('fill_method', 'inpaint'),
            kernel_size=kwargs.get('fill_kernel', 3)
        )
    
    # Step 4: Fast 3x3 median filter
    output = cv2.medianBlur(result.astype(np.float32), 3)

    return output
