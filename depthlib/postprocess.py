"""
Post-processing utilities for disparity maps.

This version is intentionally minimal and fast:
- Only left-border invalid band filling is enabled by default.
- Speckle removal / inpainting helpers are kept for future use.
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
    disp = disparity.copy().astype(np.float32)
    H, W = disp.shape

    band_widths = np.zeros(H, dtype=np.int32)
    any_band = False

    # Find band width per row
    for y in range(H):
        row = disp[y, : min(max_search, W)]
        val_idx = np.where(row != invalid_value)[0]
        if val_idx.size == 0:
            continue
        w = val_idx[0]
        if w > 0:
            band_widths[y] = w
            any_band = True

    if not any_band:
        return disp

    # Fill each detected band from the first valid value to its right
    for y in range(H):
        w = band_widths[y]
        if w == 0:
            continue
        src_val = disp[y, w]
        disp[y, :w] = src_val

    return disp


def postprocess_disparity(disparity: np.ndarray, **kwargs) -> np.ndarray:
    """
    Minimal post-processing for disparity maps.

    Currently:
    - Optionally fix left invalid band via fill_left_band.
    - No speckle, outlier, inpaint or median steps are applied by default.

    Parameters
    ----------
    disparity : np.ndarray
        Input disparity map.
    invalidate_value : float, optional
        Code used for invalid disparities (default -1.0).
    apply_fill_from_right : bool, optional
        If True, apply left-band filling.

    Returns
    -------
    result : np.ndarray
        Refined disparity map.
    """
    invalid_value = kwargs.get("invalidate_value", -1.0)
    result = disparity.copy()

    if kwargs.get("apply_fill_from_right", False):
        result = fill_left_band(result, invalid_value=invalid_value)

    return result
