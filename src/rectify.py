"""Stereo image rectification utilities."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

__all__ = ["rectify_images"]


def _parse_matrix(raw: str) -> np.ndarray:
	"""Parse a calibration matrix formatted as ``[a b c; d e f; g h i]``."""
	values = raw.strip().lstrip("[").rstrip("]")
	rows = []
	for row in values.split(";"):
		if not row.strip():
			continue
		rows.append([float(x) for x in row.strip().split()])
	matrix = np.array(rows, dtype=np.float64)
	if matrix.shape != (3, 3):
		raise ValueError(f"Expected 3x3 matrix, got shape {matrix.shape} from '{raw}'")
	return matrix


def _coerce_numeric(value: str) -> float:
	"""Convert a stringified numeric value to ``int`` when possible, else ``float``."""
	value = value.strip()
	if value.lower() in {"nan", "inf", "-inf"}:
		return float(value)
	try:
		return int(value)
	except ValueError:
		return float(value)


def _load_calibration(calib_path: Path) -> Dict[str, np.ndarray | float | int]:
	if not calib_path.exists():
		raise FileNotFoundError(f"Calibration file not found at '{calib_path}'")

	data: Dict[str, np.ndarray | float | int] = {}
	with calib_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line or line.startswith(("#", "//")):
				continue
			if "=" not in line:
				continue
			key, raw_value = [part.strip() for part in line.split("=", 1)]
			if key.lower().startswith("cam"):
				data[key.lower()] = _parse_matrix(raw_value)
			else:
				data[key.lower()] = _coerce_numeric(raw_value)

	for required in ("cam0", "cam1", "baseline", "width", "height"):
		if required not in data:
			raise KeyError(f"Missing '{required}' entry in calibration file '{calib_path}'")

	data["width"] = int(data["width"])
	data["height"] = int(data["height"])
	data["ndisp"] = int(data.get("ndisp", 0))
	data["baseline"] = float(data["baseline"])
	data["doffs"] = float(data.get("doffs", 0.0))
	return data


@lru_cache(maxsize=None)
def _load_calibration_cached(calib_path: str) -> Dict[str, np.ndarray | float | int]:
	return _load_calibration(Path(calib_path))


def _ensure_image_size(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
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
	calib_path: str | Path | None = None,
	alpha: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray | float]]:
	"""Rectify a stereo image pair using intrinsic parameters from ``calib.txt``.

	Parameters
	----------
	img_L, img_R:
		Left and right stereo images as ``numpy.ndarray``. They can be color or grayscale;
		the function returns rectified grayscale images.
	calib_path:
		Optional explicit path to the calibration file. Defaults to the project-level
		``calib.txt`` located in the repository root.
	alpha:
		Free-scaling parameter passed to :func:`cv2.stereoRectify` controlling the amount
		of valid pixels retained. ``0`` keeps the maximum useful area while cropping
		invalid regions; ``1`` keeps all original pixels.

	Returns
	-------
	rectified_L, rectified_R:
		Grayscale rectified views.
	meta:
		Dictionary containing auxiliary products such as projection matrices and the
		disparity-to-depth re-projection matrix ``Q`` for downstream stages.
	"""

	calib = _load_calibration_cached(str(calib_path))

	cam_mtx_L = np.asarray(calib["cam0"], dtype=np.float64)
	cam_mtx_R = np.asarray(calib["cam1"], dtype=np.float64)
	dist_L = np.zeros(5, dtype=np.float64)
	dist_R = np.zeros(5, dtype=np.float64)

	image_size = (calib["width"], calib["height"])

	img_L = _ensure_image_size(img_L, image_size)
	img_R = _ensure_image_size(img_R, image_size)

	# Assume cameras are translated along the x-axis by the provided baseline with
	# no relative rotation (typical for already-rectified rigs like Middlebury).
	rotation = np.eye(3, dtype=np.float64)
	translation = np.array([-calib["baseline"], 0.0, 0.0], dtype=np.float64)

	rect_L, rect_R, proj_L, proj_R, Q, roi_L, roi_R = cv2.stereoRectify(
		cam_mtx_L,
		dist_L,
		cam_mtx_R,
		dist_R,
		image_size,
		rotation,
		translation,
		flags=cv2.CALIB_ZERO_DISPARITY,
		alpha=alpha,
	)

	map1_L, map2_L = cv2.initUndistortRectifyMap(
		cam_mtx_L, dist_L, rect_L, proj_L, image_size, cv2.CV_32FC1
	)
	map1_R, map2_R = cv2.initUndistortRectifyMap(
		cam_mtx_R, dist_R, rect_R, proj_R, image_size, cv2.CV_32FC1
	)

	gray_L = _to_grayscale(img_L)
	gray_R = _to_grayscale(img_R)

	rectified_L = cv2.remap(gray_L, map1_L, map2_L, cv2.INTER_LINEAR)
	rectified_R = cv2.remap(gray_R, map1_R, map2_R, cv2.INTER_LINEAR)

	meta = {
		"proj_L": proj_L,
		"proj_R": proj_R,
		"rect_R": rect_R,
		"rect_L": rect_L,
		"Q": Q,
		"roi_L": roi_L,
		"roi_R": roi_R,
		"baseline": float(calib["baseline"]),
		"focal_length_pix": float(cam_mtx_L[0, 0]),
		"doffs": float(calib.get("doffs", 0.0)),
		"maps_L": (map1_L, map2_L),
		"maps_R": (map1_R, map2_R),
	}

	return rectified_L, rectified_R, meta