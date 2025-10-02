@echo off
echo ===================================
echo Indian Sign Language Detection
echo Original OpenCV Implementation
echo ===================================
echo.
echo Instructions:
echo 1. Position your hand clearly in front of the camera
echo 2. Show signs for letters A-Z or numbers 1-9
echo 3. The detected sign will appear on screen
echo 4. Press ESC key to quit
echo.
echo Starting camera...

REM Navigate to the script directory
cd /d "%~dp0"

REM Run the detection script
D:\isl-appp\venv\Scripts\python.exe isl_detection.py

echo.
echo Application stopped.
pause