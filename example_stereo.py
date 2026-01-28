import depthlib
import time
import numpy as np

if __name__ == "__main__":
    left_image_path = './assets/stereo_pairs/im0.png'
    right_image_path = './assets/stereo_pairs/im1.png'

    ndisp = 280
    focal_length = 3997.684
    baseline_mm = 193.001
    doffs = 131.111

    # ndisp = 128
    # focal_length = 679.01
    # baseline_mm = 572.5
    # doffs = 0

    # Using StereoDepthEstimator
    estimator = depthlib.StereoDepthEstimator(left_source=left_image_path, right_source=right_image_path, 
                                              downscale_factor=0.5)
    estimator.configure_sgbm(
        num_disp=ndisp,
        focal_length=focal_length,
        baseline=baseline_mm / 1000.0,
        doffs=doffs,
    )
    start_time = time.time()

    disparity_px, depth_m = estimator.estimate_depth()
    
    latency_ms = (time.time() - start_time) * 1000
    print(f"Depth estimation completed in {latency_ms:.2f} ms")
    estimator.visualize_results()

    # Print raw disparity statistics
    valid_disp = disparity_px > 0
    print("\n=== Raw Disparity Statistics ===")
    print(f"Disparity range: {disparity_px[valid_disp].min():.2f} - {disparity_px[valid_disp].max():.2f} pixels")
    print(f"Invalid disparities: {(~valid_disp).sum() / valid_disp.size * 100:.1f}%")