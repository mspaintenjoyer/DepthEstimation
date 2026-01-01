"""Live stereo depth demo: capture -> SGBM -> display.

Usage examples:
- Two USB cams: set left_src = 0, right_src = 1
- Two video files: set paths to files
- RTSP/URL sources: set RTSP URLs
Press ESC to exit.
"""
import time
from typing import Union

import cv2
import numpy as np

from depthlib import StereoDepthEstimator
from depthlib.input import stereo_stream


def main():
    # Configure your sources here
    left_src: Union[int, str] = './assets/left.mp4'
    right_src: Union[int, str] = './assets/right.mp4'

    downscale = 0.7  # smaller -> faster

    ndisp = 128
    focal_length = 679.01
    baseline_mm = 572.5
    doffs = 0

    estimator = StereoDepthEstimator(
        left_source=None,
        right_source=None,
        downscale_factor=downscale,
    )
    estimator.configure_sgbm(
        num_disp=ndisp,
        focal_length=focal_length,
        baseline=baseline_mm / 1000.0,
        doffs=doffs,
    )

    # Allow window resizing for local preview
    cv2.namedWindow("Depth (live)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Depth (live)", 960, 540)


    for left_frame, right_frame in stereo_stream(left_src, right_src, downscale_factor=downscale):
        t0 = time.time()
        disparity_px, depth_m = estimator.estimate_depth_frame(left_frame, right_frame)
        fps = 1.0 / max(time.time() - t0, 1e-6)

        # Colorize depth for quick display (near=warm, far=cool)
        if depth_m is not None:
            valid_depth = np.isfinite(depth_m) & (depth_m > 0)

            if valid_depth.any():
                display_max_depth_m = 50.0
                depth_clipped = np.clip(depth_m, 0, display_max_depth_m)
                depth_clipped[~valid_depth] = display_max_depth_m  # send invalid to far color

                # Emphasize near-range variation with gamma curve
                depth_ratio = depth_clipped / display_max_depth_m
                depth_gamma = np.power(depth_ratio, 0.5)
                depth_norm = (depth_gamma * 255).astype("uint8")

                depth_norm_inv = 255 - depth_norm  # flip so nearer = hotter (red/yellow)
                depth_vis = cv2.applyColorMap(depth_norm_inv, cv2.COLORMAP_TURBO)
                cv2.putText(depth_vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(depth_vis, f"Display cap: {display_max_depth_m:.0f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                depth_vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
                cv2.putText(depth_vis, "No valid depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Depth (live)", depth_vis)
        else:
            # Fallback to disparity visualization if depth could not be computed
            disp_vis = cv2.normalize(disparity_px, None, 0, 255, cv2.NORM_MINMAX)
            disp_vis = cv2.applyColorMap(disp_vis.astype("uint8"), cv2.COLORMAP_JET)
            cv2.putText(disp_vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Disparity (live)", disp_vis)

        # To stream out instead of showing locally, pipe disp_vis or left_frame into ffmpeg/rtmp here.

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
