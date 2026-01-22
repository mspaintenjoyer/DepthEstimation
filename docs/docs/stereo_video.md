
# Stereo Depth from Video

## StereoDepthEstimatorVideo

The `StereoDepthEstimatorVideo` class performs depth estimation from synchronized stereo video streams.

### Constructor

```python
StereoDepthEstimatorVideo(
    left_source: str,
    right_source: str,
    downscale_factor: float = 0.5
)
