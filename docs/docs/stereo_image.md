# Stereo Depth from Images

The `StereoDepthEstimator` class computes depth maps from rectified stereo image pairs using Semi-Global Block Matching (SGBM).

---

## Class: StereoDepthEstimator

```python
from depthlib import StereoDepthEstimator
```

### Constructor

```python
StereoDepthEstimator(
    left_source: str = None,
    right_source: str = None,
    downscale_factor: float = 1.0
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `left_source` | `str` | `None` | Path to the left stereo image |
| `right_source` | `str` | `None` | Path to the right stereo image |
| `downscale_factor` | `float` | `1.0` | Scale factor for image resizing (0 < factor ≤ 1.0). Lower values = faster processing |

!!! warning "Downscale Factor"
    The `downscale_factor` must be between 0 (exclusive) and 1.0 (inclusive). Values outside this range will raise a `ValueError`.

**Example:**

```python
estimator = StereoDepthEstimator(
    left_source='./left.png',
    right_source='./right.png',
    downscale_factor=0.5  # Process at 50% resolution
)
```

---

## Methods

### configure_sgbm()

Configure the Semi-Global Block Matching algorithm parameters.

```python
configure_sgbm(**kwargs) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_disp` | `int` | `0` | Minimum disparity value |
| `num_disp` | `int` | `128` | Number of disparities (must be divisible by 16) |
| `block_size` | `int` | `5` | Block size for matching (odd number, typically 3-11) |
| `disp12_max_diff` | `int` | `1` | Max allowed difference in left-right disparity check |
| `prefilter_cap` | `int` | `31` | Truncation value for prefiltered pixels |
| `uniqueness_ratio` | `int` | `10` | Margin (%) by which best match must win |
| `speckle_window_size` | `int` | `50` | Max size of smooth disparity regions |
| `speckle_range` | `int` | `2` | Max disparity variation within each region |
| `sgbm_mode` | `str` | `'sgbm_3way'` | Algorithm mode: `'sgbm'`, `'hh'`, `'sgbm_3way'`, `'hh4'` |
| `focal_length` | `float` | `None` | Camera focal length in pixels |
| `baseline` | `float` | `None` | Stereo baseline in meters |
| `doffs` | `float` | `0.0` | Disparity offset (cx_right - cx_left) |
| `max_depth` | `float` | `None` | Maximum depth value to clamp results |
| `hole_filling` | `bool` | `False` | Enable hole filling in disparity map |

!!! info "Automatic Scaling"
    Parameters `num_disp`, `focal_length`, and `doffs` are automatically scaled by the `downscale_factor`.

**Example:**

```python
estimator.configure_sgbm(
    num_disp=280,
    block_size=5,
    focal_length=3997.684,
    baseline=0.193,
    doffs=131.111,
    hole_filling=True
)
```

---

### get_sgbm_params()

Retrieve the current SGBM configuration.

```python
get_sgbm_params() -> Dict[str, Any]
```

**Returns:**

A dictionary containing all current SGBM parameters.

**Example:**

```python
params = estimator.get_sgbm_params()
print(f"Current num_disp: {params['num_disp']}")
```

---

### estimate_depth()

Compute disparity and depth maps from the loaded stereo pair.

```python
estimate_depth() -> Tuple[np.ndarray, Optional[np.ndarray]]
```

**Returns:**

| Return Value | Type | Description |
|--------------|------|-------------|
| `disparity_px` | `np.ndarray` | Disparity map in pixels (float32) |
| `depth_m` | `np.ndarray` or `None` | Depth map in meters (float32), or `None` if `focal_length` and `baseline` are not set |

!!! note "Pipeline"
    The estimation pipeline: Raw images → Rectification → Disparity computation → Depth mapping

**Example:**

```python
disparity_px, depth_m = estimator.estimate_depth()

# Analyze results
valid_mask = disparity_px > 0
print(f"Disparity range: {disparity_px[valid_mask].min():.1f} - {disparity_px[valid_mask].max():.1f} px")

if depth_m is not None:
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    print(f"Depth range: {depth_m[valid_depth].min():.2f} - {depth_m[valid_depth].max():.2f} m")
```

---

### visualize_results()

Display the computed disparity and depth maps using Matplotlib.

```python
visualize_results() -> None
```

!!! warning "Prerequisites"
    You must call `estimate_depth()` before calling `visualize_results()`, otherwise a `ValueError` will be raised.

**Example:**

```python
estimator.estimate_depth()
estimator.visualize_results()  # Opens matplotlib windows
```

---

## Complete Example

```python
import depthlib
import time
import numpy as np

# Configuration
left_image = './assets/stereo_pairs/im0.png'
right_image = './assets/stereo_pairs/im1.png'

# Camera parameters (from calibration)
ndisp = 280
focal_length = 3997.684  # pixels
baseline_mm = 193.001    # millimeters
doffs = 131.111          # disparity offset

# Initialize estimator
estimator = depthlib.StereoDepthEstimator(
    left_source=left_image,
    right_source=right_image,
    downscale_factor=0.5
)

# Configure SGBM
estimator.configure_sgbm(
    num_disp=ndisp,
    focal_length=focal_length,
    baseline=baseline_mm / 1000.0,  # Convert to meters
    doffs=doffs,
)

# Estimate depth with timing
start = time.time()
disparity_px, depth_m = estimator.estimate_depth()
latency_ms = (time.time() - start) * 1000
print(f"Depth estimation completed in {latency_ms:.2f} ms")

# Print statistics
valid_disp = disparity_px > 0
print(f"\n=== Disparity Statistics ===")
print(f"Range: {disparity_px[valid_disp].min():.2f} - {disparity_px[valid_disp].max():.2f} pixels")
print(f"Invalid: {(~valid_disp).sum() / valid_disp.size * 100:.1f}%")

# Visualize
estimator.visualize_results()
```

---

## See Also

- [SGBM Configuration](sgbm_config.md) - Detailed parameter tuning guide
- [Stereo Depth Video](stereo_video.md) - Real-time video depth estimation
- [Visualization Utilities](visualizations.md) - Custom visualization options
