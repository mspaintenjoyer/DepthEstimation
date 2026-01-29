import numpy as np
import pytest
from depthlib.rectify import RectificationCache

def test_rectification_caching_logic():
    """
    Ensure that calling get_maps with identical parameters returns 
    the exact same dictionary object (verification of caching).
    """
    cache = RectificationCache()
    
    # Dummy parameters
    K = np.eye(3)
    dist = np.zeros(5)
    R = np.eye(3)
    T = np.array([0.5, 0, 0])
    
    # First Call: Should compute and store
    maps_1 = cache.get_maps(
        cam_matrix_L=K, cam_matrix_R=K,
        baseline=0.5, image_width=640, image_height=480,
        dist_coeff_L=dist, dist_coeff_R=dist,
        rotation=R, translation=T
    )
    
    # Second Call: Should retrieve from cache
    maps_2 = cache.get_maps(
        cam_matrix_L=K, cam_matrix_R=K,
        baseline=0.5, image_width=640, image_height=480,
        dist_coeff_L=dist, dist_coeff_R=dist,
        rotation=R, translation=T
    )
    
   
    assert maps_1 is not None
    assert maps_2 is not None
    #check
    assert maps_1 is maps_2 
    
    # Modify params slightly -> Should return NEW object
    maps_3 = cache.get_maps(
        cam_matrix_L=K, cam_matrix_R=K,
        baseline=0.6, # Changed baseline
        image_width=640, image_height=480,
        dist_coeff_L=dist, dist_coeff_R=dist,
        rotation=R, translation=T
    )
    
    assert maps_1 is not maps_3