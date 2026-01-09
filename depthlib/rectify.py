"""Stereo image rectification utilities."""

from __future__ import annotations

import warnings
from typing import Tuple

import cv2
import numpy as np

__all__ = ["rectify_images"]


def _ensure_image_size(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
	"""Ensure image matches the expected size, resizing if necessary."""
	height, width = image.shape[:2]
	expected_width, expected_height = size
	if (width, height) == size:
		return image

	warnings.warn(
		"Input image size %sx%s does not match calibration %sx%s; resizing for rectification." %
		(width, height, expected_width, expected_height),
		RuntimeWarning,
		stacklevel=2,
	)
	return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
	"""Convert image to grayscale if needed."""
	if image.ndim == 2:
		return image
	if image.ndim == 3 and image.shape[2] == 1:
		return image[:, :, 0]
	if image.ndim == 3 and image.shape[2] == 3:
		try:
			return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		except cv2.error:
			return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	raise ValueError("Unsupported image format for grayscale conversion")


def rectify_images(
	img_L: np.ndarray,
	img_R: np.ndarray,
	cam_matrix_L: np.ndarray,
	cam_matrix_R: np.ndarray,
	baseline: float,
	image_width: int,
	image_height: int,
	dist_coeff_L: np.ndarray | None = None,
	dist_coeff_R: np.ndarray | None = None,
	rotation: np.ndarray | None = None,
	translation: np.ndarray | None = None,
	alpha: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Rectify a stereo image pair using calibration parameters.

	Parameters
	----------
	img_L : np.ndarray
		Left stereo image. Can be color or grayscale.
	img_R : np.ndarray
		Right stereo image. Can be color or grayscale.
	cam_matrix_L : np.ndarray
		3x3 camera matrix for left camera.
	cam_matrix_R : np.ndarray
		3x3 camera matrix for right camera.
	baseline : float
		Baseline distance between cameras (in meters or consistent units).
	image_width : int
		Expected image width in pixels.
	image_height : int
		Expected image height in pixels.
	dist_coeff_L : np.ndarray, optional
		Distortion coefficients for left camera (k1, k2, p1, p2, k3).
	dist_coeff_R : np.ndarray, optional
		Distortion coefficients for right camera (k1, k2, p1, p2, k3).
	rotation : np.ndarray, optional
		3x3 rotation matrix from left to right camera.
	translation : np.ndarray, optional
		3x1 translation vector from left to right camera (meters).
	alpha : float, optional
		Free-scaling parameter (0.0-1.0). 0 crops to valid pixels only,
		1 keeps all original pixels. Default is 0.0.

	Returns
	-------
	rectified_L : np.ndarray
		Rectified left image (grayscale).
	rectified_R : np.ndarray
		Rectified right image (grayscale).
	"""
	# Ensure matrices are float64
	cam_mtx_L = np.asarray(cam_matrix_L, dtype=np.float64)
	cam_mtx_R = np.asarray(cam_matrix_R, dtype=np.float64)

	# Use provided distortion or default to zero-distortion
	default_dist = np.zeros(5, dtype=np.float64)
	dist_L = np.asarray(dist_coeff_L, dtype=np.float64) if dist_coeff_L is not None else default_dist
	dist_R = np.asarray(dist_coeff_R, dtype=np.float64) if dist_coeff_R is not None else default_dist

	image_size = (image_width, image_height)

	# Ensure images match expected size
	img_L = _ensure_image_size(img_L, image_size)
	img_R = _ensure_image_size(img_R, image_size)

	# Use provided extrinsics or fall back to canonical baseline along x-axis
	R = np.asarray(rotation, dtype=np.float64) if rotation is not None else np.eye(3, dtype=np.float64)
	T = np.asarray(translation, dtype=np.float64) if translation is not None else np.array([-baseline, 0.0, 0.0], dtype=np.float64)

	# Compute stereo rectification
	rect_L, rect_R, proj_L, proj_R, Q, roi_L, roi_R = cv2.stereoRectify(
		cam_mtx_L,
		dist_L,
		cam_mtx_R,
		dist_R,
		image_size,
		R,
		T,
		flags=cv2.CALIB_ZERO_DISPARITY,
		alpha=alpha,
	)

	# Generate rectification maps
	map1_L, map2_L = cv2.initUndistortRectifyMap(
		cam_mtx_L, dist_L, rect_L, proj_L, image_size, cv2.CV_32FC1
	)
	map1_R, map2_R = cv2.initUndistortRectifyMap(
		cam_mtx_R, dist_R, rect_R, proj_R, image_size, cv2.CV_32FC1
	)

	# Convert to grayscale
	gray_L = _to_grayscale(img_L)
	gray_R = _to_grayscale(img_R)

	# Apply rectification
	rectified_L = cv2.remap(gray_L, map1_L, map2_L, cv2.INTER_LINEAR)
	rectified_R = cv2.remap(gray_R, map1_R, map2_R, cv2.INTER_LINEAR)

	return rectified_L, rectified_R