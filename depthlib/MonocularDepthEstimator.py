from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
from depthlib.visualizations import visualize_depth
import numpy as np
from typing import Literal

class MonocularDepthEstimator:
    def __init__(self, model_path, device: Literal['cpu', 'cuda']='cpu', downscale_factor: float=1.0):
        if not model_path:
            raise ValueError("Model path must be provided.")
        
        status = self._get_torch_status()
        if not status["installed"]:
            raise ImportError("PyTorch is not installed. Please install the cpu or cuda version of PyTorch.")
        
        if device == 'cuda' and not status["cuda_available"]:
            raise EnvironmentError("CUDA is not available. Please check if you have torch cuda version or use device='cpu'.")

        self.torch = status["torch"]
        self.model_path = model_path
        self.device = device
        self.downscale_factor = downscale_factor
        self.model, self.processor = None, None
        self.load_model()
        self.depth_map = None

    def load_model(self):
        # Load the pre-trained monocular depth estimation model from the specified path
        print(f"Loading model from {self.model_path}")
        
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_path, use_fast=True)
            model = AutoModelForDepthEstimation.from_pretrained(self.model_path)
            model.eval().to(self.device)
            self.model = model
            self.processor = processor
            self.warmup()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def warmup(self):
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded properly.")

        # Perform a warmup inference to optimize performance
        print("Warming up the model")
        
        dummy_image = Image.new('RGB', (224, 224), color = 'white')
        inputs = self.processor(images=dummy_image, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            _ = self.model(**inputs)

    def estimate_depth(self, image_path):
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded properly.")

        # Estimate depth from the given image using the loaded model
        print("Estimating depth for the provided image")
        
        image = Image.open(image_path).convert("RGB")
        if self.downscale_factor != 1.0:
            new_size = (int(image.width * self.downscale_factor), int(image.height * self.downscale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with self.torch.no_grad():
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

    def _get_torch_status(self):
        try:
            import torch
        except ImportError:
            return {
                "installed": False,
                "cuda_built": False,
                "cuda_available": False,
                "torch": None,
            }

        return {
            "installed": True,
            "cuda_available": torch.cuda.is_available(),
            "torch": torch,
        }