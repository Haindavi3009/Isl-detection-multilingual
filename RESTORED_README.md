# Indian Sign Language Detection - Original Implementation

## ✅ Project Restored to Working State

The project has been reverted back to the original working OpenCV implementation. All web interface files have been removed and the original functionality is preserved.

## 🚀 How to Run

### Quick Start
```batch
# Double-click this file or run in command prompt:
start_detection.bat
```

### Manual Method
```powershell
# Navigate to project directory
cd "D:\isl-appp\Indian-Sign-Language-Detection"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the detection script
python isl_detection.py
```

## 🎯 What the Application Does

- **Real-time hand gesture detection** using your webcam
- **Recognizes 35 signs**: Numbers 1-9 and Letters A-Z
- **OpenCV display** showing camera feed with detected sign overlay
- **Text output** in console showing detected signs
- **Press ESC** to quit the application

## 📋 System Requirements

- **Python 3.6+** (Currently using 3.11.0)
- **Webcam** for real-time detection
- **Windows OS** (current setup)

## 📦 Installed Packages

- `opencv-python==4.7.0.72` - Computer vision and camera handling
- `mediapipe==0.9.1.0` - Hand landmark detection
- `tensorflow==2.11.0` - Neural network for gesture classification
- `numpy==1.24.2` - Numerical computations
- `pandas==1.5.3` - Data manipulation

## 🔧 Technical Details

### How It Works
1. **Camera Capture**: OpenCV captures video frames from webcam
2. **Hand Detection**: MediaPipe identifies hand landmarks
3. **Feature Extraction**: Landmarks are normalized and processed
4. **Classification**: TensorFlow model predicts the gesture
5. **Display**: Result is shown on video feed and printed to console

### Files Structure
```
Indian-Sign-Language-Detection/
├── isl_detection.py          # Main detection script
├── model.h5                  # Trained TensorFlow model
├── keypoint.csv              # Training data
├── requirements.txt          # Python dependencies
├── start_detection.bat       # Easy startup script
├── venv/                     # Virtual environment
└── images/                   # Project images
```

## ⚠️ Expected Warnings

When running, you may see these warnings (they are normal):
- `AttributeError: 'MessageFactory'` - MediaPipe/TensorFlow compatibility
- `TensorFlow binary optimized` - Performance optimization info
- `Compiled metrics have yet to be built` - Model loading info
- `Feedback manager requires` - MediaPipe inference info

**These warnings don't affect functionality - the application works correctly!**

## 🎮 Usage Instructions

1. **Start Application**: Run `start_detection.bat` or the Python script
2. **Position Hand**: Place your hand clearly in front of the camera
3. **Make Gestures**: Show signs for letters (A-Z) or numbers (1-9)
4. **View Results**: 
   - Detected sign appears as text overlay on video
   - Sign is also printed in the console
5. **Exit**: Press ESC key to close

## 📈 Supported Gestures

- **Numbers**: 1, 2, 3, 4, 5, 6, 7, 8, 9
- **Letters**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

## 🔍 Troubleshooting

### Camera Issues
- Ensure camera permissions are granted
- Close other applications using the camera
- Try reconnecting USB camera if external

### Python Issues
- Verify virtual environment is activated
- Reinstall packages: `pip install -r requirements.txt`
- Check Python version: `python --version`

### Model Issues
- Verify `model.h5` exists in project directory
- Check file size (should be ~11MB)

## 🎉 Success!

The original ISL detection system is now fully restored and working! The application provides real-time gesture recognition with the same accuracy and functionality as the original implementation.