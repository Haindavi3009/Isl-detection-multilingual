#!/usr/bin/env python3
"""
Run the original ISL Detection script
This script starts the OpenCV-based ISL detection application
"""

import subprocess
import sys
import os

def main():
    print("=" * 50)
    print("Indian Sign Language Detection")
    print("Original OpenCV Implementation")
    print("=" * 50)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("model.h5"):
        print("ERROR: model.h5 not found!")
        print("Please run this script from the project directory")
        return 1
    
    if not os.path.exists("isl_detection.py"):
        print("ERROR: isl_detection.py not found!")
        return 1
    
    print("Starting ISL Detection...")
    print("Instructions:")
    print("1. Position your hand clearly in front of the camera")
    print("2. Show signs for letters A-Z or numbers 1-9")
    print("3. Press ESC key to quit")
    print()
    print("Starting camera in 3 seconds...")
    
    # Import time for countdown
    import time
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Camera starting now!")
    
    # Run the original detection script
    try:
        # Get the Python executable path from the virtual environment
        python_path = os.path.join("venv", "Scripts", "python.exe")
        if not os.path.exists(python_path):
            python_path = sys.executable
        
        # Run the original script
        result = subprocess.run([python_path, "isl_detection.py"], 
                              capture_output=False, 
                              text=True)
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return 0
    except Exception as e:
        print(f"Error running ISL detection: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())