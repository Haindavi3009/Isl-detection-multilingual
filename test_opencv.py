#!/usr/bin/env python3

import cv2
import numpy as np

print("Testing OpenCV installation...")
print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")

# Test basic functionality
try:
    print("Testing basic cv2 import...")
    print(f"cv2 module: {cv2}")
    print(f"cvtColor available: {hasattr(cv2, 'cvtColor')}")
    
    # Create test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    print(f"Created test image with shape: {test_image.shape}")
    
    # Test cvtColor
    print("Testing cv2.cvtColor...")
    rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    print(f"SUCCESS: cvtColor worked! Output shape: {rgb_image.shape}")
    
    # Test color constants
    print(f"COLOR_BGR2RGB constant: {cv2.COLOR_BGR2RGB}")
    
    print("All tests passed!")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()