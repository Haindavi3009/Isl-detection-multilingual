🧠 Indian Sign Language Detection & Translation (A–Z & 1–9 | Multilingual + Bidirectional)

🚀 Overview

This project detects Indian Sign Language (ISL) gestures for A–Z alphabets and 1–9 numbers using Mediapipe and a Feedforward Neural Network (FNN).

Enhancements added:

1.Sign → Text → Voice output
2.Multilingual support (8 Indian languages)
3.Text → Sign conversion for single letters and numbers
Note: Only individual letters and digits are supported; full words or sentences are not recognized.
**Note:** All output screenshots for this project are provided in the `output_screenshots` folder.
____________________________________________________________________________________________________________________________
✨ Key Features

🖐️ Detects ISL alphabets (A–Z) and numbers (1–9) in real-time using webcam
🔤 Converts detected gestures to text
🔊 Generates voice output in selected language
🌐 Supports 8 Indian languages: English, Hindi, Telugu, Malayalam, Tamil, Kannada, Bengali, Marathi
🔁 Text → Sign conversion (letters and digits only)
🎥 Real-time detection with Mediapipe and OpenCV
🧠 Built using Feedforward Neural Network (FNN) for classification
_______________________________________________________________________________________________________________________________
🌐 Supported Languages

The system supports both sign-to-text/voice and text-to-sign for the following 8 Indian languages:

English
Hindi
Telugu
Malayalam
Tamil
Kannada
Bengali
Marathi
_______________________________________________________________________________________________________________________________
🏗️ Project Structure

ISL_Multilingual_Translator/
│
├── README.md                     
├── isl_detection.py               → Detect ISL alphabets/numbers (Sign → Text/Voice)
├── text_to_sign.py                → Convert single letters/numbers to ISL gestures
├── dataset_keypoint_generation.py → Convert dataset images to hand landmarks
├── ISL_classifier.ipynb           → Train FNN model
├── model.h5                       → Trained classifier model
├── keypoint.csv                   → Hand landmark dataset
│
├── utils/
│   ├── translator.py              → Multilingual translation
│   └── tts_engine.py              → Text-to-speech
│
└── requirements.txt               
_______________________________________________________________________________________________________________________________

⚙️ Installation

Ensure Python 3.6+ is installed, then install dependencies:
pip install mediapipe opencv-python numpy tensorflow googletrans==4.0.0-rc1 gTTS playsound

(Optional: use virtual environment)

python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows
_______________________________________________________________________________________________________________________________
▶️ Usage

Sign → Text + Voice
python isl_detection.py


1.Open webcam
2.Show A–Z or 1–9 gestures
3.Recognized letter/number is displayed and spoken
4.Press ‘q’ to exit

Text → Sign
python text_to_sign.py


Enter a single letter or number
Corresponding ISL gesture is displayed
_______________________________________________________________________________________________________________________________


🧩 How It Works

1.Mediapipe detects 21 hand landmarks per frame.
2.Landmarks passed to Feedforward Neural Network (FNN) trained on ISL alphabets & numbers.
3.Predicted class (letter or number) displayed as text.
4.Text translated to selected language using Google Translate.
5.Voice generated via gTTS.
6.Text → Sign shows ISL gesture for entered letter or number.
_______________________________________________________________________________________________________________________________
🧱 Model Details

1.Model: Feedforward Neural Network (FNN)
2.Input: 42 hand keypoints (x, y coordinates)
3.Output: 36 classes (A–Z + 1–9)
4.Frameworks: TensorFlow + Mediapipe
_______________________________________________________________________________________________________________________________
🚧 Future Improvements

1.Support for full words/sentences
2.Add more gesture variations for numbers/letters
3.Develop mobile/web app
4.GUI for easier interaction
_______________________________________________________________________________________________________________________________
👩‍💻 Author

Sira Haindavi
GitHub: https://github.com/<your-username>

Email:haindavisira@gmail.com