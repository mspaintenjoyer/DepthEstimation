import numpy as np
from depthlib.stereo_core import StereoCore

def test_fast_mode_toggle():
    """
    Verify that fast_mode=True produces different output than fast_mode=False
    (implying that post-processing steps are being skipped/applied).
    """
    # Create a dummy noisy disparity map
    # We simulate a "step" function with noise
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, :50] = 50
    img[:, 50:] = 200
    # Add random noise
    noisy_img = img + np.random.randint(-10, 10, (100, 100)).astype(np.uint8)
    
    # 1. Run in FAST mode
    core_fast = StereoCore(fast_mode=True)
    # Mock the internal compute to return our noisy image
    core_fast.compute_disparity = lambda l, r: noisy_img.astype(np.float32)
    # Mock params to avoid crop errors
    core_fast.sgbm_params['num_disp'] = 0 
    
    disp_fast, _ = core_fast._process_pair(noisy_img, noisy_img)
    
    # 2. Run in SLOW (Quality) mode
    core_slow = StereoCore(fast_mode=False)
    core_slow.compute_disparity = lambda l, r: noisy_img.astype(np.float32)
    core_slow.sgbm_params['num_disp'] = 0
    
    disp_slow, _ = core_slow._process_pair(noisy_img, noisy_img)
    
    # Assertions
    # The output should NOT be identical
    assert not np.array_equal(disp_fast, disp_slow)
    
    # The slow mode (with filtering) should be "smoother"
    grad_fast = np.std(np.diff(disp_fast))
    grad_slow = np.std(np.diff(disp_slow))
    
    # Filtered image should have less high-frequency noise
    assert grad_slow < grad_fast