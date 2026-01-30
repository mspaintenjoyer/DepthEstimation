# SGBM Configuration Guide

This guide explains how to tune Semi-Global Block Matching (SGBM) parameters for optimal depth estimation results.

---

## Overview

SGBM (Semi-Global Block Matching) is the core algorithm used for stereo matching. Proper configuration is essential for accurate depth maps.

### Basic Formula

Depth is calculated from disparity using:

$$Z = \frac{f \cdot B}{d + d_{offs}}$$

Where:

- $Z$ = Depth (meters)
- $f$ = Focal length (pixels)
- $B$ = Baseline (meters)
- $d$ = Disparity (pixels)
- $d_{offs}$ = Disparity offset

---

## Camera Parameters

These parameters define your stereo camera setup and are **required for metric depth**.

### focal_length

Camera focal length in **pixels**.

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `None` |
| Units | pixels |

**How to obtain:**

```python
# From camera matrix K
K = [[fx, 0, cx],
     [0, fy, cy],
     [0, 0,  1]]
focal_length = fx  # or fy (usually similar)
```

---

### baseline

Distance between the two camera centers in **meters**.

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `None` |
| Units | meters |

**Example:**

```python
baseline_mm = 572.5  # millimeters
baseline = baseline_mm / 1000.0  # Convert to meters: 0.5725
```

---

### doffs

Disparity offset (difference in principal points between cameras).

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.0` |
| Units | pixels |

**Formula:** `doffs = cx_right - cx_left`

---

### max_depth

Maximum depth value to clamp results.

| Property | Value |
|----------|-------|
| Type | `float` or `None` |
| Default | `None` |
| Units | meters |

**Example:**

```python
estimator.configure_sgbm(
    max_depth=50.0  # Clamp depth to 50 meters
)
```

---

## Disparity Parameters

### num_disp

Number of disparities to search (disparity range).

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `128` |
| Constraint | Must be divisible by 16 |

!!! tip "Choosing num_disp"
    - **Larger values** → Detect closer objects, but slower
    - **Smaller values** → Faster, but miss close objects
    
    Rule of thumb: `num_disp ≈ image_width / 4` to `image_width / 8`

**Common values:** `64`, `128`, `192`, `256`, `280`

---

### min_disp

Minimum disparity value (where to start searching).

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `0` |

**Use cases:**

- `min_disp=0` → Objects at infinity are valid
- `min_disp>0` → Ignore very far objects

---

### block_size

Size of the matching block (window).

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `5` |
| Range | `1` to `11` (odd numbers) |

!!! tip "Choosing block_size"
    - **Smaller (3-5)** → More detail, but noisier
    - **Larger (7-11)** → Smoother, but less detail

---

## Quality Parameters

### uniqueness_ratio

Margin (%) by which the best match must win over the second-best.

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `10` |
| Range | `5` to `15` |

**Effect:**

- **Higher values** → Fewer false matches, but more holes
- **Lower values** → More matches, but potentially more errors

---

### disp12_max_diff

Maximum allowed difference in left-right disparity consistency check.

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `1` |

**Effect:**

- `1` → Strict consistency (recommended)
- `-1` → Disable left-right check

---

### prefilter_cap

Truncation value for prefiltered image pixels.

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `31` |
| Range | `1` to `63` |

---

## Speckle Filtering

Removes small isolated disparity regions (noise).

### speckle_window_size

Maximum size of smooth disparity regions to consider as valid.

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `50` |

**Effect:**

- **Larger values** → More aggressive noise removal
- `0` → Disable speckle filtering

---

### speckle_range

Maximum disparity variation within each connected component.

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `2` |

---

## Algorithm Mode

### sgbm_mode

SGBM algorithm variant to use.

| Property | Value |
|----------|-------|
| Type | `str` |
| Default | `'sgbm_3way'` |
| Options | `'sgbm'`, `'hh'`, `'sgbm_3way'`, `'hh4'` |

**Mode comparison:**

| Mode | Quality | Speed | Description |
|------|---------|-------|-------------|
| `'sgbm'` | Good | Medium | Standard 8-direction SGBM |
| `'hh'` | Good | **Fast** | Full-scale 2-pass dynamic programming |
| `'sgbm_3way'` | **Best** | Slow | 3-direction SGBM (highest quality) |
| `'hh4'` | Good | Fast | 4-direction variant |

---

## Post-Processing

### hole_filling

Enable hole filling in the disparity map.

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `False` |

**Effect:** Fills invalid disparity regions using inpainting.

---

## Configuration Examples

### High Quality (Slow)

```python
estimator.configure_sgbm(
    num_disp=256,
    block_size=5,
    uniqueness_ratio=10,
    speckle_window_size=100,
    speckle_range=2,
    sgbm_mode='sgbm_3way',
    hole_filling=True,
    focal_length=3997.684,
    baseline=0.193,
    doffs=131.111
)
```

### Balanced

```python
estimator.configure_sgbm(
    num_disp=128,
    block_size=5,
    uniqueness_ratio=10,
    speckle_window_size=50,
    sgbm_mode='sgbm_3way',
    focal_length=679.01,
    baseline=0.5725
)
```

### Fast (Real-time)

```python
estimator.configure_sgbm(
    num_disp=64,
    block_size=7,
    uniqueness_ratio=5,
    speckle_window_size=0,  # Disable
    sgbm_mode='hh',  # Fastest mode
    focal_length=679.01,
    baseline=0.5725
)
```

### Close-Range Scene

```python
estimator.configure_sgbm(
    num_disp=280,  # Large range for close objects
    block_size=3,  # Small blocks for detail
    max_depth=5.0,  # Limit to 5 meters
    focal_length=3997.684,
    baseline=0.193
)
```

---

## Troubleshooting

### Too Many Holes

- Decrease `uniqueness_ratio`
- Enable `hole_filling=True`
- Increase `speckle_window_size`

### Noisy Disparity

- Increase `block_size`
- Increase `speckle_window_size` and `speckle_range`
- Use `sgbm_mode='sgbm_3way'`

### Missing Close Objects

- Increase `num_disp`

### Depth Values Too Large/Small

- Verify `focal_length`, `baseline`, and `doffs` values
- Check that `baseline` is in **meters**
- Verify `focal_length` is in **pixels** (not mm)

---

## Parameter Reference Table

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `focal_length` | `float` | `None` | Focal length (pixels) |
| `baseline` | `float` | `None` | Camera baseline (meters) |
| `doffs` | `float` | `0.0` | Disparity offset (pixels) |
| `max_depth` | `float` | `None` | Max depth clamp (meters) |
| `num_disp` | `int` | `128` | Disparity range |
| `min_disp` | `int` | `0` | Minimum disparity |
| `block_size` | `int` | `5` | Matching window size |
| `uniqueness_ratio` | `int` | `10` | Match uniqueness (%) |
| `disp12_max_diff` | `int` | `1` | L-R consistency threshold |
| `prefilter_cap` | `int` | `31` | Prefilter truncation |
| `speckle_window_size` | `int` | `50` | Speckle filter size |
| `speckle_range` | `int` | `2` | Speckle disparity range |
| `sgbm_mode` | `str` | `'sgbm_3way'` | Algorithm variant |
| `hole_filling` | `bool` | `False` | Enable hole filling |
