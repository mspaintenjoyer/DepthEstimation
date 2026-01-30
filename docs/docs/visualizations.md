# Visualization Utilities

Depthlib provides built-in visualization functions for displaying stereo pairs, disparity maps, and depth maps.

---

## Available Functions

```python
from depthlib import (
    visualize_stereo_pair,
    visualize_disparity,
    visualize_depth,
    visualize_disparity_and_depth
)
```

---

## visualize_stereo_pair()

Display two images (stereo pair or rectified pair) side by side.

```python
visualize_stereo_pair(
    left_img_rgb: np.ndarray,
    right_img_rgb: np.ndarray,
    title_left: str = 'Left Image',
    title_right: str = 'Right Image'
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `left_img_rgb` | `np.ndarray` | *Required* | Left image (RGB or grayscale) |
| `right_img_rgb` | `np.ndarray` | *Required* | Right image (RGB or grayscale) |
| `title_left` | `str` | `'Left Image'` | Title for left image |
| `title_right` | `str` | `'Right Image'` | Title for right image |

**Example:**

```python
import cv2
from depthlib import visualize_stereo_pair

left = cv2.imread('left.png')
right = cv2.imread('right.png')

# Convert BGR to RGB for display
left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

visualize_stereo_pair(left_rgb, right_rgb)
```

---

## visualize_disparity()

Visualize a disparity map with proper colormap.

```python
visualize_disparity(
    disparity_px: np.ndarray,
    title: str = 'Disparity Map',
    cmap: str = 'jet',
    vmin: float = None,
    vmax: float = None
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disparity_px` | `np.ndarray` | *Required* | Raw disparity map in pixels |
| `title` | `str` | `'Disparity Map'` | Plot title |
| `cmap` | `str` | `'jet'` | Matplotlib colormap |
| `vmin` | `float` | `None` | Min value for color scaling (auto if None) |
| `vmax` | `float` | `None` | Max value for color scaling (auto if None) |

**Available colormaps:**

- `'jet'` - Rainbow (default)
- `'turbo'` - Improved rainbow
- `'plasma'` - Purple to yellow
- `'viridis'` - Green to yellow
- `'magma'` - Black to white through purple

**Example:**

```python
from depthlib import visualize_disparity

# After computing disparity
visualize_disparity(disparity_px, title='My Disparity', cmap='turbo')

# With custom range
visualize_disparity(disparity_px, vmin=10, vmax=200)
```

---

## visualize_depth()

Visualize a depth map with proper colormap.

```python
visualize_depth(
    depth_m: np.ndarray,
    title: str = 'Depth Map',
    cmap: str = 'turbo_r',
    max_depth: float = None,
    show_invalid: bool = True,
    show_meter: bool = True
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth_m` | `np.ndarray` | *Required* | Depth map in meters |
| `title` | `str` | `'Depth Map'` | Plot title |
| `cmap` | `str` | `'turbo_r'` | Matplotlib colormap (reversed for depth) |
| `max_depth` | `float` | `None` | Max depth to display (auto if None) |
| `show_invalid` | `bool` | `True` | Show invalid regions (inf/far) in black |
| `show_meter` | `bool` | `True` | Display depth in meters on colorbar |

!!! tip "Color Interpretation"
    With `'turbo_r'` colormap:
    
    - 🔴 Red/Yellow = Close objects
    - 🔵 Blue = Far objects

**Example:**

```python
from depthlib import visualize_depth

# Basic usage
visualize_depth(depth_m)

# Custom settings
visualize_depth(
    depth_m,
    title='Scene Depth',
    max_depth=10.0,  # Limit display to 10 meters
    cmap='plasma'
)
```

---

## visualize_disparity_and_depth()

Visualize disparity and depth side by side with optional reference image.

```python
visualize_disparity_and_depth(
    disparity_px: np.ndarray,
    depth_m: np.ndarray,
    left_img: np.ndarray = None
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `disparity_px` | `np.ndarray` | *Required* | Disparity map in pixels |
| `depth_m` | `np.ndarray` | *Required* | Depth map in meters |
| `left_img` | `np.ndarray` | `None` | Optional reference image |

**Example:**

```python
from depthlib import visualize_disparity_and_depth

# Without reference image
visualize_disparity_and_depth(disparity_px, depth_m)

# With reference image
visualize_disparity_and_depth(disparity_px, depth_m, left_img=left_image)
```

---

## Live Visualization

For real-time video depth estimation, the library provides OpenCV-based visualization:

### visualize_depth_live()

Display depth map in a live window with FPS overlay (color mode).

```python
visualize_depth_live(depth_m: np.ndarray, fps: float) -> None
```

**Features:**

- Real-time display using OpenCV
- FPS counter overlay
- Display cap indicator (50m default)
- Turbo colormap (red = close, blue = far)

---

### visualize_depth_live_gray()

Display depth map in a live window with FPS overlay (grayscale mode).

```python
visualize_depth_live_gray(depth_m: np.ndarray, fps: float) -> None
```

**Features:**

- Grayscale display (white = close, black = far)
- FPS counter overlay
- Display cap indicator

---

## Usage with Estimators

### StereoDepthEstimator

```python
import depthlib

estimator = depthlib.StereoDepthEstimator(
    left_source='left.png',
    right_source='right.png'
)
estimator.configure_sgbm(num_disp=128, focal_length=1000, baseline=0.1)

# Estimate depth
disparity_px, depth_m = estimator.estimate_depth()

# Use built-in visualization
estimator.visualize_results()

# Or use custom visualization
depthlib.visualize_depth(depth_m, title='Custom Title', max_depth=20.0)
```

### StereoDepthEstimatorVideo

```python
from depthlib import StereoDepthEstimatorVideo

estimator = StereoDepthEstimatorVideo(
    left_source='left.mp4',
    right_source='right.mp4',
    visualize_live=True,      # Enable live visualization
    visualize_gray=False      # Use color (True for grayscale)
)

for depth_m in estimator.estimate_depth():
    # Visualization is handled automatically
    pass
```

### MonocularDepthEstimator

```python
import depthlib

estimator = depthlib.MonocularDepthEstimator(
    model_path='path/to/model',
    device='cuda'
)

depth_map = estimator.estimate_depth('image.png')

# Use built-in visualization
estimator.visualize_depth()

# Or use custom visualization (note: show_meter=False for relative depth)
depthlib.visualize_depth(depth_map, show_meter=False)
```

---

## Colormap Reference

| Colormap | Best For | Description |
|----------|----------|-------------|
| `'jet'` | Disparity | Classic rainbow |
| `'turbo'` | Disparity | Improved rainbow |
| `'turbo_r'` | Depth | Reversed (close=hot) |
| `'plasma'` | General | Purple to yellow |
| `'viridis'` | General | Perceptually uniform |
| `'magma'` | General | Dark to light |
| `'gray'` | Simple | Grayscale |

---

## Tips

1. **Depth colormap**: Use reversed colormaps (`_r` suffix) for depth to show close objects as warm colors

2. **Auto-scaling**: Leave `vmin`/`vmax` as `None` for automatic range detection using percentiles

3. **Invalid values**: Depth maps may contain `inf` values for invalid regions; these are handled automatically

4. **Live performance**: `visualize_depth_live()` uses OpenCV for faster rendering than Matplotlib
