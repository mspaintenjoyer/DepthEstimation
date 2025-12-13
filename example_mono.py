import depthlib
import time

if __name__ == "__main__":
    left_image_path = './assets/im0.png'
    right_image_path = './assets/im1.png'

    #Using MonocularDepthEstimator
    model_path = "models/hub/models--depth-anything--Depth-Anything-V2-Base-hf/snapshots/b1958afc87fb45a9e3746cb387596094de553ed8"
    estimator = depthlib.MonocularDepthEstimator(model_path=model_path)

    start_time = time.time()

    depth_map = estimator.estimate_depth(image_path=left_image_path)

    latency_ms = (time.time() - start_time) * 1000
    print(f"Depth estimation completed in {latency_ms:.2f} ms")

    estimator.visualize_depth()