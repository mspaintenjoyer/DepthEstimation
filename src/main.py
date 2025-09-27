from SimpleSGBM import run_sgbm_stage
from input import load_stereo_pair, visualize_stereo_pair
from rectify import rectify_images

if __name__ == "__main__":
    left_image_path = '../assets/stereo_pairs/im0.png'
    right_image_path = '../assets/stereo_pairs/im1.png'

    left_img_rgb, right_img_rgb = load_stereo_pair(left_image_path, right_image_path)
    visualize_stereo_pair(left_img_rgb, right_img_rgb)

    rectified_L, rectified_R, calib_data = rectify_images(left_img_rgb, right_img_rgb, calib_path='../assets/calib.txt')
    visualize_stereo_pair(rectified_L, rectified_R)

    disparity_px, depth_m = run_sgbm_stage(
        rectified_L, rectified_R,
        f_pixels=calib_data.get("focal_length_pix"),
        baseline_m=calib_data.get("baseline")/1000.0
    )
    visualize_stereo_pair(disparity_px / disparity_px.max(), depth_m / (depth_m.max() + 1e-6))
