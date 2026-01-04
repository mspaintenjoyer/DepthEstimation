import time
import numpy as np
import depthlib
from depthlib.calibration import load_middlebury_calib



if __name__ == "__main__":
    left_image_path = "./assets/stereo_pairs/im0.png"
    right_image_path = "./assets/stereo_pairs/im1.png"

    ndisp = 256          # before downscale; will be scaled by 0.5 → 128
    focal_length = 3997.684
    baseline_mm = 193.001
    doffs = 131.111

    cal = load_middlebury_calib("./assets/calib.txt")

    estimator = depthlib.StereoDepthEstimator(left_source=left_image_path, right_source=right_image_path, downscale_factor=0.5)
    estimator.configure_sgbm(
        num_disp=cal.ndisp,
        block_size=5,
        uniqueness_ratio=10,
        focal_length=float(cal.K0[0, 0]),
        baseline=float(cal.baseline_m),
        doffs=float(cal.doffs),
        image_width=int(cal.width * 0.5),
        image_height=int(cal.height * 0.5),
    )

    start_time = time.time()
    disparity_px, depth_m = estimator.estimate_depth()
    latency_ms = (time.time() - start_time) * 1000.0
    print(f"Depth estimation completed in {latency_ms:.2f} ms")

    estimator.visualize_results()

    invalid_value = -1.0
    valid_disp = disparity_px > invalid_value

    print("\n=== Raw Disparity Statistics ===")
    print(
        f"Disparity range: {disparity_px[valid_disp].min():.2f} - "
        f"{disparity_px[valid_disp].max():.2f} pixels"
    )
    print(f"Invalid disparities: {(~valid_disp).sum() / valid_disp.size * 100:.2f}%")
