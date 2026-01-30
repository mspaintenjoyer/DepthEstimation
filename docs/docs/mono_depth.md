# Monocular Depth Estimation

The `MonocularDepthEstimator` class estimates depth from a single RGB image using a pre-trained deep learning model (Depth Anything V2).

!!! note "No Stereo Required"
    Unlike stereo depth estimation, monocular depth works with a single image. However, the depth values are **relative** (not metric) and depend on model generalization.

---

## Class: MonocularDepthEstimator

```python
from depthlib import MonocularDepthEstimator
```

### Constructor

```python
MonocularDepthEstimator(
    model_path: str,
    device: Literal['cpu', 'cuda'] = 'cpu',
    downscale_factor: float = 1.0
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | *Required* | Path to the pre-trained model directory |
| `device` | `str` | `'cpu'` | Computation device: `'cpu'` or `'cuda'` (GPU) |
| `downscale_factor` | `float` | `1.0` | Scale factor for image resizing (0 < factor ≤ 1.0) |

!!! warning "Requirements"
    - **PyTorch** must be installed
    - For `device='cuda'`, PyTorch CUDA version is required
    - Model files must be downloaded separately

**Example:**

```python
model_path = "models/hub/models--depth-anything--Depth-Anything-V2-Base-hf/snapshots/b1958afc..."

estimator = MonocularDepthEstimator(
    model_path=model_path,
    device='cuda',  # Use GPU
    downscale_factor=0.5
)
```

---

## Methods

### estimate_depth()

Estimate relative depth from a single image.

```python
estimate_depth(image_path: str) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_path` | `str` | Path to the input RGB image |

**Returns:**

| Return Value | Type | Description |
|--------------|------|-------------|
| `depth_map` | `np.ndarray` | Relative depth map (higher values = closer) |

!!! info "Depth Values"
    The returned depth values are **inverted** for visualization purposes:
    - Higher values = closer objects
    - Lower values = farther objects
    
    Values are relative, not metric (not in meters).

**Example:**

```python
depth_map = estimator.estimate_depth(image_path='./image.png')

print(f"Depth map shape: {depth_map.shape}")
print(f"Value range: {depth_map.min():.2f} - {depth_map.max():.2f}")
```

---

### visualize_depth()

Display the estimated depth map using Matplotlib.

```python
visualize_depth() -> None
```

!!! warning "Prerequisites"
    You must call `estimate_depth()` before calling `visualize_depth()`, otherwise a `RuntimeError` will be raised.

**Example:**

```python
estimator.estimate_depth('./image.png')
estimator.visualize_depth()  # Opens matplotlib window
```

---

### load_model()

Load or reload the pre-trained model.

```python
load_model() -> None
```

!!! note "Automatic Loading"
    This method is called automatically during initialization. You only need to call it manually if you want to reload the model.

---

### warmup()

Perform a warmup inference to optimize performance.

```python
warmup() -> None
```

!!! note "Automatic Warmup"
    This method is called automatically during initialization.

---

## Model Setup

### Supported Models

The library supports **Depth Anything V2** models from Hugging Face:

| Model | Size | Quality | Speed |
|-------|------|---------|-------|
| `Depth-Anything-V2-Small-hf` | ~98MB | Good | Fast |
| `Depth-Anything-V2-Base-hf` | ~390MB | Better | Medium |
| `Depth-Anything-V2-Large-hf` | ~1.4GB | Best | Slow |

### Download Model

Download the model from Hugging Face Hub:

```bash
# Using git-lfs
git lfs install
git clone https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf

# Or using huggingface_hub
pip install huggingface_hub
huggingface-cli download depth-anything/Depth-Anything-V2-Base-hf
```

### Model Directory Structure

```
models/hub/models--depth-anything--Depth-Anything-V2-Base-hf/
└── snapshots/
    └── b1958afc87fb45a9e3746cb387596094de553ed8/
        ├── config.json
        ├── model.safetensors
        └── preprocessor_config.json
```

---

## Complete Example

```python
import depthlib
import time

# Model path
model_path = "models/hub/models--depth-anything--Depth-Anything-V2-Base-hf/snapshots/b1958afc87fb45a9e3746cb387596094de553ed8"

# Initialize estimator
estimator = depthlib.MonocularDepthEstimator(
    model_path=model_path,
    device='cuda',      # Use GPU for faster inference
    downscale_factor=0.5
)

# Estimate depth
image_path = './assets/image.png'

start_time = time.time()
depth_map = estimator.estimate_depth(image_path=image_path)
latency_ms = (time.time() - start_time) * 1000

print(f"Depth estimation completed in {latency_ms:.2f} ms")
print(f"Depth map shape: {depth_map.shape}")
print(f"Value range: {depth_map.min():.2f} - {depth_map.max():.2f}")

# Visualize
estimator.visualize_depth()
```

---

## Error Handling

### Common Errors

**PyTorch Not Installed:**
```python
# Raises ImportError
ImportError: PyTorch is not installed. Please install the cpu or cuda version of PyTorch.
```

**CUDA Not Available:**
```python
# Raises EnvironmentError when device='cuda' but CUDA is not available
EnvironmentError: CUDA is not available. Please check if you have torch cuda version or use device='cpu'.
```

**Model Not Found:**
```python
# Raises Exception when model_path is invalid
Exception: Error loading model: ...
```

**No Model Path:**
```python
# Raises ValueError
ValueError: Model path must be provided.
```

---

## Performance Tips

1. **Use GPU**: Set `device='cuda'` for significantly faster inference
2. **Downscale Images**: Use `downscale_factor=0.5` or lower for faster processing
3. **Batch Processing**: The model performs a warmup on first run; subsequent calls are faster

---

## Monocular vs Stereo Depth

| Feature | Monocular | Stereo |
|---------|-----------|--------|
| Input | Single image | Image pair |
| Output | Relative depth | Metric depth (meters) |
| Calibration | Not required | Required |
| Accuracy | Depends on scene | Geometric precision |
| Speed | Model-dependent | Fast (CPU-based) |

---

## See Also

- [Stereo Depth Images](stereo_image.md) - Metric depth from stereo pairs
- [Stereo Depth Video](stereo_video.md) - Real-time video depth
- [Visualization Utilities](visualizations.md) - Custom visualization options
