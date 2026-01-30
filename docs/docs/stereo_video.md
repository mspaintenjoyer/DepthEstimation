# Stereo Depth from Video

The `StereoDepthEstimatorVideo` class performs real-time depth estimation from synchronized stereo video streams or live cameras.

---

## Class: StereoDepthEstimatorVideo

```python
from depthlib import StereoDepthEstimatorVideo
```

### Constructor

```python
StereoDepthEstimatorVideo(
    left_source: Union[int, str] = None,
    right_source: Union[int, str] = None,
    downscale_factor: float = 1.0,
    visualize_live: bool = False,
    saving_path: str = None,
    fast_mode: bool = False,
    use_threading: bool = True,
    target_fps: int = 30,
    drop_frames: bool = False,
    visualize_gray: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `left_source` | `int` or `str` | `None` | Left video path, camera index (e.g., `0`), or RTSP URL |
| `right_source` | `int` or `str` | `None` | Right video path, camera index (e.g., `1`), or RTSP URL |
| `downscale_factor` | `float` | `1.0` | Scale factor for frame resizing (0 < factor ≤ 1.0) |
| `visualize_live` | `bool` | `False` | Show live depth visualization window |
| `saving_path` | `str` | `None` | Path to save output video (not yet implemented) |
| `fast_mode` | `bool` | `False` | Enable fast mode for higher FPS (skips postprocessing) |
| `use_threading` | `bool` | `True` | Use threaded frame capture for better FPS |
| `target_fps` | `int` | `30` | Maximum FPS to process (0 = unlimited) |
| `drop_frames` | `bool` | `False` | Drop frames when processing is slow (recommended for live cameras) |
| `visualize_gray` | `bool` | `False` | Use grayscale visualization instead of color |

**Source Types:**

| Type | Example | Description |
|------|---------|-------------|
| Camera Index | `0`, `1` | USB camera device number |
| Video File | `'./video.mp4'` | Path to video file |
| RTSP URL | `'rtsp://...'` | Network stream URL |

**Example:**

```python
# From video files
estimator = StereoDepthEstimatorVideo(
    left_source='./left.mp4',
    right_source='./right.mp4',
    downscale_factor=0.7,
    visualize_live=True,
    target_fps=30
)

# From USB cameras
estimator = StereoDepthEstimatorVideo(
    left_source=0,
    right_source=1,
    downscale_factor=0.5,
    visualize_live=True,
    drop_frames=True  # Recommended for live cameras
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

**Example:**

```python
estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725,  # 572.5mm in meters
    doffs=0,
    hole_filling=True
)
```

---

### estimate_depth()

Start processing the video stream and yield depth maps for each frame.

```python
estimate_depth() -> Generator[np.ndarray, None, None]
```

**Yields:**

| Value | Type | Description |
|-------|------|-------------|
| `depth_m` | `np.ndarray` | Depth map in meters (float32) for each frame |

!!! info "Generator Pattern"
    This method returns a generator that yields depth maps frame-by-frame. The generator handles video capture, processing, and cleanup automatically.

!!! tip "Keyboard Control"
    Press **ESC** to stop the video processing loop.

**Example:**

```python
for depth_m in estimator.estimate_depth():
    # Process each depth frame
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    if valid_depth.any():
        min_depth = depth_m[valid_depth].min()
        print(f"Closest object: {min_depth:.2f} m")
```

---

## Performance Options

### Fast Mode

Enable `fast_mode=True` to skip expensive postprocessing steps:

```python
estimator = StereoDepthEstimatorVideo(
    left_source=0,
    right_source=1,
    fast_mode=True,  # Faster but noisier depth
    downscale_factor=0.5
)
```

**Trade-offs:**

| Setting | Quality | Speed |
|---------|---------|-------|
| `fast_mode=False` | Higher | Slower |
| `fast_mode=True` | Lower | Faster |

### Threading

Threading improves FPS by capturing frames in a separate thread:

```python
# Threaded capture (default, recommended)
estimator = StereoDepthEstimatorVideo(
    left_source=0,
    right_source=1,
    use_threading=True
)

# Non-threaded (useful for debugging)
estimator = StereoDepthEstimatorVideo(
    left_source='./left.mp4',
    right_source='./right.mp4',
    use_threading=False
)
```

### Frame Dropping

For live cameras, enable frame dropping to prevent lag:

```python
estimator = StereoDepthEstimatorVideo(
    left_source=0,
    right_source=1,
    drop_frames=True  # Drop old frames when processing is slow
)
```

!!! warning "Video Files"
    Set `drop_frames=False` when processing video files to ensure all frames are processed.

---

## Complete Examples

### Live USB Cameras

```python
from depthlib import StereoDepthEstimatorVideo

# Two USB cameras
estimator = StereoDepthEstimatorVideo(
    left_source=0,
    right_source=1,
    downscale_factor=0.5,
    visualize_live=True,
    fast_mode=True,
    drop_frames=True,
    target_fps=30
)

estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725
)

# Press ESC to exit
for depth_m in estimator.estimate_depth():
    pass
```

### Video Files

```python
from depthlib import StereoDepthEstimatorVideo

estimator = StereoDepthEstimatorVideo(
    left_source='./assets/left.mp4',
    right_source='./assets/right.mp4',
    downscale_factor=0.7,
    visualize_live=True,
    use_threading=True,
    drop_frames=False,  # Process all frames
    target_fps=30
)

estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725,
    hole_filling=True
)

frame_count = 0
for depth_m in estimator.estimate_depth():
    frame_count += 1

print(f"Processed {frame_count} frames")
```

### Grayscale Visualization

```python
estimator = StereoDepthEstimatorVideo(
    left_source='./left.mp4',
    right_source='./right.mp4',
    visualize_live=True,
    visualize_gray=True  # Grayscale depth display
)
```

---

## Visualization

When `visualize_live=True`, a window displays the depth map with:

- **FPS counter** - Current processing rate
- **Display cap** - Maximum displayed depth (50m)
- **Color coding** (default):
    - 🔴 Red/Yellow = Close objects
    - 🔵 Blue = Far objects
- **Grayscale mode** (`visualize_gray=True`):
    - ⬜ White = Close objects
    - ⬛ Black = Far objects

---

## See Also

- [SGBM Configuration](sgbm_config.md) - Detailed parameter tuning guide
- [Stereo Depth Images](stereo_image.md) - Single image pair depth estimation
- [Visualization Utilities](visualizations.md) - Custom visualization options
