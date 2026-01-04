import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../src/Monocular/main.py -> parents[2] == repo root

# Hugging Face cache / model directory under project root
HF_MODELS_DIR = PROJECT_ROOT / "models"
HF_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Set before importing transformers so it picks up the path early.
os.environ["HF_HOME"] = str(HF_MODELS_DIR)

from transformers import pipeline  # noqa: E402  (import after setting env var)

# Instantiate pipeline (will download/cache to PROJECT_ROOT/models if needed)
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")

# Resolve image path relative to project root for robustness
image_path = PROJECT_ROOT / "assets" / "stereo_pairs" / "im0.png"
if not image_path.exists():
	raise FileNotFoundError(f"Input image not found at: {image_path}")

depth_map = pipe(str(image_path))["depth"]

# Convert to numpy array and invert so closer objects have higher values
depth_map = np.array(depth_map)
depth_map = np.max(depth_map) - depth_map  # Invert: closer = higher value

plt.imshow(depth_map, cmap="plasma")
plt.axis("off")
plt.show()

