from input import load_stereo_pair, visualize_stereo_pair

if __name__ == "__main__":
    left_image_path = '../assets/stereo_pairs/im0.png'
    right_image_path = '../assets/stereo_pairs/im1.png'

    left_img_rgb, right_img_rgb = load_stereo_pair(left_image_path, right_image_path)
    visualize_stereo_pair(left_img_rgb, right_img_rgb)
