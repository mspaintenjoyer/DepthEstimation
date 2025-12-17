"""
Rectification utilities.

Current version assumes the user provides already-rectified pairs unless
calibration matrices are set in the estimator.
"""

from typing import Tuple

import cv2
import numpy as np


def rectify_images(
    left_img: np.ndarray,
    right_img: np.ndarray,
    cam_matrix_L: np.ndarray,
    cam_matrix_R: np.ndarray,
    baseline_m: float,
    img_width: int,
    img_height: int,
    alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectify left/right images using provided calibration.

    Notes
    -----
    For now this is a placeholder: it simply resizes/crops to the given
    dimensions and assumes input images are already rectified. To be replaced 
    with a full cv2.stereoRectify + initUndistortRectifyMap pipeline once
    we have complete calibration (R, T, distCoeffs, etc.).
    """
    left_resized = cv2.resize(left_img, (img_width, img_height), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_img, (img_width, img_height), interpolation=cv2.INTER_AREA)
    return left_resized, right_resized
