# Examples

This page provides complete code examples for common use cases.

---

## Stereo Image Depth Estimation

### Basic Example

```python
import depthlib

# Load stereo pair and estimate depth
estimator = depthlib.StereoDepthEstimator(
    left_source='./assets/stereo_pairs/im0.png',
    right_source='./assets/stereo_pairs/im1.png',
    downscale_factor=0.5
)

# Configure with camera parameters
estimator.configure_sgbm(
    num_disp=280,
    focal_length=3997.684,
    baseline=0.193,  # meters
    doffs=131.111
)

# Estimate and visualize
disparity_px, depth_m = estimator.estimate_depth()
estimator.visualize_results()
```

### With Statistics

```python
import depthlib
import numpy as np
import time

# Setup
estimator = depthlib.StereoDepthEstimator(
    left_source='./left.png',
    right_source='./right.png',
    downscale_factor=0.5
)

estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725
)

# Measure timing
start = time.time()
disparity_px, depth_m = estimator.estimate_depth()
elapsed_ms = (time.time() - start) * 1000

# Print statistics
print(f"Processing time: {elapsed_ms:.2f} ms")

# Disparity stats
valid_disp = disparity_px > 0
print(f"\n=== Disparity ===")
print(f"Range: {disparity_px[valid_disp].min():.1f} - {disparity_px[valid_disp].max():.1f} px")
print(f"Invalid pixels: {(~valid_disp).sum() / valid_disp.size * 100:.1f}%")

# Depth stats
if depth_m is not None:
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    print(f"\n=== Depth ===")
    print(f"Range: {depth_m[valid_depth].min():.2f} - {depth_m[valid_depth].max():.2f} m")
    print(f"Mean: {depth_m[valid_depth].mean():.2f} m")

estimator.visualize_results()
```

---

## Live Video Depth Estimation

### USB Cameras

```python
from depthlib import StereoDepthEstimatorVideo

# Two USB cameras (indices 0 and 1)
estimator = StereoDepthEstimatorVideo(
    left_source=0,
    right_source=1,
    downscale_factor=0.5,
    visualize_live=True,
    fast_mode=True,      # Faster processing
    drop_frames=True,    # Don't queue old frames
    target_fps=30
)

estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725,
    hole_filling=True
)

print("Press ESC to exit")
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
    drop_frames=False,   # Process all frames
    target_fps=30
)

estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725
)

frame_count = 0
for depth_m in estimator.estimate_depth():
    frame_count += 1

print(f"Processed {frame_count} frames")
```

### Save Closest Distance

```python
from depthlib import StereoDepthEstimatorVideo
import numpy as np

estimator = StereoDepthEstimatorVideo(
    left_source='./left.mp4',
    right_source='./right.mp4',
    visualize_live=True
)

estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725
)

closest_distances = []

for depth_m in estimator.estimate_depth():
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if valid.any():
        closest = depth_m[valid].min()
        closest_distances.append(closest)
        print(f"Closest object: {closest:.2f} m")

# Analyze results
if closest_distances:
    print(f"\n=== Summary ===")
    print(f"Min distance seen: {min(closest_distances):.2f} m")
    print(f"Max distance seen: {max(closest_distances):.2f} m")
```

---

## Monocular Depth Estimation

### Basic Example

```python
import depthlib

model_path = "models/hub/models--depth-anything--Depth-Anything-V2-Base-hf/snapshots/b1958afc87fb45a9e3746cb387596094de553ed8"

estimator = depthlib.MonocularDepthEstimator(
    model_path=model_path,
    device='cuda',
    downscale_factor=0.5
)

depth_map = estimator.estimate_depth('./image.png')
estimator.visualize_depth()
```

### Batch Processing

```python
import depthlib
import os
import time

model_path = "path/to/model"
image_dir = "./images"
output_dir = "./depth_maps"

os.makedirs(output_dir, exist_ok=True)

# Initialize once
estimator = depthlib.MonocularDepthEstimator(
    model_path=model_path,
    device='cuda',
    downscale_factor=1.0
)

# Process all images
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]

for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    
    start = time.time()
    depth_map = estimator.estimate_depth(image_path)
    elapsed = (time.time() - start) * 1000
    
    print(f"{filename}: {elapsed:.1f} ms")
    
    # Save depth map
    import numpy as np
    output_path = os.path.join(output_dir, f"depth_{filename}")
    np.save(output_path.replace('.png', '.npy').replace('.jpg', '.npy'), depth_map)
```

---

## Custom Visualization

### Side-by-Side Comparison

```python
import depthlib
import matplotlib.pyplot as plt
import numpy as np

# Estimate depth
estimator = depthlib.StereoDepthEstimator(
    left_source='./left.png',
    right_source='./right.png'
)
estimator.configure_sgbm(num_disp=128, focal_length=1000, baseline=0.1)
disparity_px, depth_m = estimator.estimate_depth()

# Custom visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
import cv2
left = cv2.imread('./left.png')
left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
axes[0].imshow(left_rgb)
axes[0].set_title('Left Image')
axes[0].axis('off')

# Disparity
valid_disp = disparity_px > 0
im1 = axes[1].imshow(disparity_px, cmap='jet', 
                      vmin=np.percentile(disparity_px[valid_disp], 1),
                      vmax=np.percentile(disparity_px[valid_disp], 99))
axes[1].set_title('Disparity (pixels)')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Depth
if depth_m is not None:
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    max_depth = np.percentile(depth_m[valid_depth], 95)
    depth_display = np.clip(depth_m, 0, max_depth)
    depth_display[~valid_depth] = max_depth
    
    im2 = axes[2].imshow(depth_display, cmap='turbo_r', vmin=0, vmax=max_depth)
    axes[2].set_title('Depth (meters)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
plt.savefig('depth_comparison.png', dpi=150)
plt.show()
```

---

## Parameter Tuning

### Finding Optimal num_disp

```python
import depthlib
import numpy as np
import matplotlib.pyplot as plt

# Test different num_disp values
num_disp_values = [64, 128, 192, 256]
results = []

for nd in num_disp_values:
    estimator = depthlib.StereoDepthEstimator(
        left_source='./left.png',
        right_source='./right.png',
        downscale_factor=0.5
    )
    estimator.configure_sgbm(
        num_disp=nd,
        focal_length=1000,
        baseline=0.1
    )
    
    import time
    start = time.time()
    disparity_px, depth_m = estimator.estimate_depth()
    elapsed = (time.time() - start) * 1000
    
    valid = disparity_px > 0
    coverage = valid.sum() / valid.size * 100
    
    results.append({
        'num_disp': nd,
        'time_ms': elapsed,
        'coverage': coverage
    })
    
    print(f"num_disp={nd}: {elapsed:.1f}ms, coverage={coverage:.1f}%")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

nds = [r['num_disp'] for r in results]
ax1.bar(nds, [r['time_ms'] for r in results])
ax1.set_xlabel('num_disp')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Processing Time')

ax2.bar(nds, [r['coverage'] for r in results])
ax2.set_xlabel('num_disp')
ax2.set_ylabel('Valid pixels (%)')
ax2.set_title('Disparity Coverage')

plt.tight_layout()
plt.show()
```

---

## Error Handling

### Robust Stereo Estimation

```python
import depthlib
import numpy as np

def estimate_depth_safe(left_path, right_path, **sgbm_params):
    """Estimate depth with error handling."""
    try:
        estimator = depthlib.StereoDepthEstimator(
            left_source=left_path,
            right_source=right_path,
            downscale_factor=0.5
        )
        estimator.configure_sgbm(**sgbm_params)
        
        disparity_px, depth_m = estimator.estimate_depth()
        
        # Validate results
        valid_disp = disparity_px > 0
        if valid_disp.sum() < 0.1 * valid_disp.size:
            print("Warning: Less than 10% valid disparities")
            return None, None
        
        return disparity_px, depth_m
        
    except FileNotFoundError as e:
        print(f"Image not found: {e}")
        return None, None
    except ValueError as e:
        print(f"Invalid parameter: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

# Usage
disparity, depth = estimate_depth_safe(
    './left.png',
    './right.png',
    num_disp=128,
    focal_length=1000,
    baseline=0.1
)

if depth is not None:
    print("Success!")
```

---

## Integration Example

### Obstacle Detection

```python
from depthlib import StereoDepthEstimatorVideo
import numpy as np

DANGER_DISTANCE = 1.0  # meters
WARNING_DISTANCE = 3.0  # meters

estimator = StereoDepthEstimatorVideo(
    left_source=0,
    right_source=1,
    visualize_live=True,
    fast_mode=True
)

estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725
)

print("Obstacle Detection Running... (ESC to exit)")

for depth_m in estimator.estimate_depth():
    valid = np.isfinite(depth_m) & (depth_m > 0)
    
    if valid.any():
        closest = depth_m[valid].min()
        
        if closest < DANGER_DISTANCE:
            print(f"⚠️  DANGER! Object at {closest:.2f}m")
        elif closest < WARNING_DISTANCE:
            print(f"⚡ Warning: Object at {closest:.2f}m")
```
