import numpy as np
import cv2

"""
Pipeline Integration (Data Flow):
  Preprocess Module -> Rectify/Undistort -> [This Module: SGBM Disparity]
  -> Postprocess (e.g., WLS filter, left-right check)
  -> Depth Mapping (Z = f*B/d)
  -> Visualization / Point Cloud / Tracking

Note :
  This SGBM module explicitly follows from the Rectified Pair module and
  assumes inputs are rectified grayscale images (same size & epipolar-aligned).
  Downstream modules that will be integrated:
    - Postprocess: speckle filtering, WLS guided filter, LR-consistency
    - Depth: disparity-to-depth conversion (metric)
    - Vizualisation/3D: overlay, point-cloud generation, mesh or odometry
"""

def build_sgbm(
    min_disp=0, num_disp=128, block_size=5,
    disp12_max_diff=1, prefilter_cap=31, uniqueness_ratio=10,
    speckle_window_size=50, speckle_range=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
):
    """
    Build StereoSGBM matcher.
    
    Note: num_disp must be divisible by 16 for SGBM.
    If using calibration file, set num_disp from 'ndisp' parameter.
    """
    channels = 1
    P1 = 8 * channels * (block_size ** 2)
    P2 = 32 * channels * (block_size ** 2)
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12_max_diff,
        preFilterCap=prefilter_cap,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=mode,
    )

def compute_disparity(rectified_L, rectified_R, sgbm):
    # Inputs come from Rectify module
    disp_fixed = sgbm.compute(rectified_L, rectified_R)
    return disp_fixed.astype(np.float32) / 16.0

def disparity_to_depth(disp, f_pixels, baseline_m, doffs=0.0, eps=1e-6, max_depth=None):
    """
    Convert disparity to depth using the formula: Z = (f * B) / (d - doffs)
    
    Parameters:
    -----------
    disp : np.ndarray
        Disparity map in pixels
    f_pixels : float
        Focal length in pixels
    baseline_m : float
        Baseline distance in meters
    doffs : float
        Disparity offset (difference in principal points between cameras)
    eps : float
        Minimum valid disparity threshold (default 1e-6 is too strict)
    max_depth : float, optional
        Maximum depth value in meters. Values beyond this are clamped.
        If None, no clamping is applied.
    
    Returns:
    --------
    Z : np.ndarray
        Depth map in meters. Invalid regions are set to inf (not 0) to distinguish
        from actual measurements.
    """
    # Adjust disparity by offset before depth calculation
    adjusted_disp = disp - doffs
    
    # Calculate depth, using inf for invalid disparities
    # This avoids division by zero and maintains distinction between
    # "no measurement" (inf) and "measured as close" (small value)
    Z = np.full_like(disp, np.inf, dtype=np.float32)
    valid_mask = adjusted_disp > eps
    Z[valid_mask] = (f_pixels * baseline_m) / adjusted_disp[valid_mask]
    
    # Optionally clamp to maximum depth
    if max_depth is not None:
        Z[Z > max_depth] = max_depth
    
    return Z

def run_sgbm_stage(rectified_L, rectified_R, f_pixels=None, baseline_m=None, doffs=0.0, 
                   min_disparity=0.5, max_depth=100.0, sgbm=None):
    """
    Entrypoint for pipeline stage:
      Inputs: rectified_L, rectified_R (grayscale) from Rectify module
      Outputs:
        - disparity_px (float32): to Postprocess module 
        - depth_m (float32, optional): to Viz/3D module (point cloud) if f & B provided
    
    Parameters:
    -----------
    doffs : float
        Disparity offset from calibration (accounts for principal point differences)
    min_disparity : float
        Minimum valid disparity in pixels (default 0.5). Disparities below this
        are considered invalid (too far or no match).
    max_depth : float
        Maximum depth to compute in meters (default 100m). Helps clamp far regions.
    """
    if sgbm is None:
        sgbm = build_sgbm()
    disparity_px = compute_disparity(rectified_L, rectified_R, sgbm)

    depth_m = None
    if f_pixels is not None and baseline_m is not None:
        depth_m = disparity_to_depth(disparity_px, f_pixels, baseline_m, doffs, 
                                     eps=min_disparity, max_depth=max_depth)

    return disparity_px, depth_m
