import depthlib

if __name__ == "__main__":
    left_image_path = './assets/stereo_pairs/im0.png'
    right_image_path = './assets/stereo_pairs/im1.png'

    ndisp = 280  
    focal_length = 3997.684
    baseline_mm = 193.001
    doffs = 131.111

    # Using StereoDepthEstimator
    estimator = depthlib.StereoDepthEstimator(left_source=left_image_path, right_source=right_image_path)
    estimator.configure_sgbm(
        num_disp=ndisp,
        focal_length=focal_length,
        baseline=baseline_mm / 1000.0,
        doffs=doffs
    )
    disparity_px, depth_m = estimator.estimate_depth()

    estimator.visualize_results()

    # Print raw disparity statistics
    valid_disp = disparity_px > 0
    print("\n=== Raw Disparity Statistics ===")
    print(f"Disparity range: {disparity_px[valid_disp].min():.2f} - {disparity_px[valid_disp].max():.2f} pixels")
    print(f"Invalid disparities: {(~valid_disp).sum() / valid_disp.size * 100:.1f}%")