# Monocular Depth Estimation

## MonoDepthEstimator

The `MonoDepthEstimator` class estimates depth from a single RGB image using a pretrained deep learning model.

This approach does not require stereo image pairs, but depth accuracy depends on model generalization and scene characteristics.

### Constructor

```python
MonoDepthEstimator(
    model_name: str = "Intel/dpt-large",
    device: str = "cpu"
)
