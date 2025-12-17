from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import torch
from depthlib.visualizations import visualize_depth
import numpy as np

class MonocularDepthEstimator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model, self.processor = self.load_model()
        self.depth_map = None

    def load_model(self):
        # Load the pre-trained monocular depth estimation model from the specified path
        print(f"Loading model from {self.model_path}")
        
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_path, use_fast=True)
            model = AutoModelForDepthEstimation.from_pretrained(self.model_path)
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

        return model, processor

    def estimate_depth(self, image_path):
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded properly.")

        # Estimate depth from the given image using the loaded model
        print("Estimating depth for the provided image")
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        predicted_depth = np.array(predicted_depth.squeeze().cpu())
        predicted_depth = np.max(predicted_depth) - predicted_depth  # Invert depth for better visualization

        self.depth_map = predicted_depth
        return predicted_depth
    
    def visualize_depth(self):
        if self.depth_map is None:
            raise RuntimeError("Depth map is not available. Please run estimate_depth first.")

        print("Visualizing depth map")
        visualize_depth(self.depth_map,)