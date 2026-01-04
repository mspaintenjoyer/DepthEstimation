"""Input utilities for loading stereo image pairs."""

import cv2


def load_stereo_pair(left_image_path, right_image_path, downscale_factor=1.0):
    """
    Load a stereo image pair from file paths.
    
    Parameters:
    -----------
    left_image_path : str
        Path to the left image
    right_image_path : str
        Path to the right image
    
    Returns:
    --------
    left_img_rgb : np.ndarray
        Left image in RGB format
    right_img_rgb : np.ndarray
        Right image in RGB format
    """
    # Load images
    left_img = cv2.imread(left_image_path)
    right_img = cv2.imread(right_image_path)

    if left_img is None or right_img is None:
        raise FileNotFoundError("One or both image paths are invalid.")

    # Convert from BGR (OpenCV default) to RGB for Matplotlib display
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    # Downscale if needed
    if downscale_factor != 1.0:
        new_size_left = (int(left_img_rgb.shape[1] * downscale_factor), int(left_img_rgb.shape[0] * downscale_factor))
        new_size_right = (int(right_img_rgb.shape[1] * downscale_factor), int(right_img_rgb.shape[0] * downscale_factor))
        left_img_rgb = cv2.resize(left_img_rgb, new_size_left, interpolation=cv2.INTER_AREA)
        right_img_rgb = cv2.resize(right_img_rgb, new_size_right, interpolation=cv2.INTER_AREA)

    return left_img_rgb, right_img_rgb