# Stereo Depth from Images

## StereoDepthEstimator

The `StereoDepthEstimator` class estimates depth from a pair of rectified stereo images.

### Constructor

```python
StereoDepthEstimator(
    left_source: str,
    right_source: str,
    downscale_factor: float = 0.5
)
