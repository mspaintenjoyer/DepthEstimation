"""Input utilities for loading stereo image pairs and live streams."""

from typing import Generator, Iterable, Tuple, Union

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


    return left_img_rgb, right_img_rgb


# --- Live video helpers ---

def open_capture(source: Union[int, str]) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture from camera index, file path, or URL."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return cap


def _read_frame(cap: cv2.VideoCapture, downscale_factor: float) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from video source")
    if downscale_factor != 1.0:
        new_size = (
            int(frame.shape[1] * downscale_factor),
            int(frame.shape[0] * downscale_factor),
        )
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame


def stereo_stream(
    left_source: Union[int, str],
    right_source: Union[int, str],
    downscale_factor: float = 1.0,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield synchronized frames from two captures.

    The generator raises StopIteration when either stream ends. Caller is
    responsible for releasing captures when finished.
    """
    if downscale_factor <= 0 or downscale_factor > 1.0:
        raise ValueError("downscale_factor must be between 0 and 1.")

    cap_L = open_capture(left_source)
    cap_R = open_capture(right_source)

    try:
        while True:
            left = _read_frame(cap_L, downscale_factor)
            right = _read_frame(cap_R, downscale_factor)
            yield left, right
    finally:
        cap_L.release()
        cap_R.release()
