"""
Input utilities: loading stereo image pairs.
"""

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


def load_image(path: Union[str, Path], downscale_factor: float = 1.0) -> np.ndarray:
    """
    Load an image as BGR uint8 and optionally downscale.

    Parameters
    ----------
    path : str or Path
        Image file path.
    downscale_factor : float
        Factor in (0, 1]. Values < 1 downscale the image.

    Returns
    -------
    img : np.ndarray
        Loaded image (H x W x 3, BGR, uint8).
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if downscale_factor != 1.0:
        h, w = img.shape[:2]
        new_size = (int(w * downscale_factor), int(h * downscale_factor))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    return img


def load_stereo_pair(
    left_source: Union[str, Path],
    right_source: Union[str, Path],
    downscale_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a stereo pair as BGR images with optional downscaling.

    Parameters
    ----------
    left_source, right_source : str or Path
        Left/right image paths.
    downscale_factor : float
        Downscale factor for both images.

    Returns
    -------
    left_img, right_img : np.ndarray
        Loaded left and right images (BGR, uint8).
    """
    left = load_image(left_source, downscale_factor)
    right = load_image(right_source, downscale_factor)
    return left, right
