import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, make_response
import base64
from PIL import Image
import io
import copy
import itertools
import logging
import sys
import requests
import urllib.parse # <-- IMPORT ADDED HERE
import tempfile
import soundfile as sf

# Redirect stderr to suppress TensorFlow startup messages
class DevNull:
    def write(self, msg):
        pass
    def flush(self):
        pass

# Suppress TensorFlow warnings and messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error
tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

app = Flask(__name__)

# Add route to serve images from the images folder
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('images', filename)

# Configure Flask logger to only show errors
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# You have correctly moved the dictionaries here. This is perfect.
language_mappings = {
    'en-US': { 
        '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine',
        'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M',
        'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'
    },
    'te-IN': { 
        '1': 'ఒకటి', '2': 'రెండు', '3': 'మూడు', '4': 'నాలుగు', '5': 'ఐదు', '6': 'ఆరు', '7': 'ఏడు', '8': 'ఎనిమిది', '9': 'తొమ్మిది',
        'A': 'ఎ', 'B': 'బీ', 'C': 'సీ', 'D': 'డీ', 'E': 'ఈ', 'F': 'ఎఫ్', 'G': 'జీ', 'H': 'ఎచ్', 'I': 'ఐ', 'J': 'జే', 'K': 'కే', 'L': 'ఎల్', 'M': 'ఎం',
        'N': 'ఎన్', 'O': 'ఓ', 'P': 'పీ', 'Q': 'క్యూ', 'R': 'ఆర్', 'S': 'ఎస్', 'T': 'టీ', 'U': 'యూ', 'V': 'వీ', 'W': 'డబ్ల్యూ', 'X': 'ఎక్స్', 'Y': 'వై', 'Z': 'జెడ్'
    },
    'hi-IN': { 
        '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार', '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ',
        'A': 'ए', 'B': 'बी', 'C': 'सी', 'D': 'डी', 'E': 'ई', 'F': 'एफ', 'G': 'जी', 'H': 'एच', 'I': 'आई', 'J': 'जे', 'K': 'के', 'L': 'एल', 'M': 'एम',
        'N': 'एन', 'O': 'ओ', 'P': 'पी', 'Q': 'क्यू', 'R': 'आर', 'S': 'एस', 'T': 'टी', 'U': 'यू', 'V': 'वी', 'W': 'डब्ल्यू', 'X': 'एक्स', 'Y': 'वाई', 'Z': 'जेड'
    },
    'ml-IN': { 
        '1': 'ഒന്ന്', '2': 'രണ്ട്', '3': 'മൂന്ന്', '4': 'നാല്', '5': 'അഞ്ച്', '6': 'ആറ്', '7': 'ഏഴ്', '8': 'എട്ട്', '9': 'ഒമ്പത്',
        'A': 'എ', 'B': 'ബി', 'C': 'സി', 'D': 'ഡി', 'E': 'ഇ', 'F': 'എഫ്', 'G': 'ജി', 'H': 'എച്ച്', 'I': 'ഐ', 'J': 'ജെ', 'K': 'കെ', 'L': 'എല്', 'M': 'എം',
        'N': 'എന്', 'O': 'ഓ', 'P': 'പി', 'Q': 'ക്യു', 'R': 'ആര്', 'S': 'എസ്', 'T': 'ടി', 'U': 'യു', 'V': 'വി', 'W': 'ഡബ്ല്യു', 'X': 'എക്സ്', 'Y': 'വൈ', 'Z': 'സെഡ്'
    },
    'ta-IN': { 
        '1': 'ஒன்று', '2': 'இரண்டு', '3': 'மூன்று', '4': 'நான்கு', '5': 'ஐந்து', '6': 'ஆறு', '7': 'ஏழு', '8': 'எட்டு', '9': 'ஒன்பது',
        'A': 'ஏ', 'B': 'பி', 'C': 'சி', 'D': 'டி', 'E': 'ஈ', 'F': 'எஃப்', 'G': 'ஜி', 'H': 'எச்', 'I': 'ஐ', 'J': 'ஜே', 'K': 'கே', 'L': 'எல்', 'M': 'எம்',
        'N': 'என்', 'O': 'ஓ', 'P': 'பி', 'Q': 'க்யூ', 'R': 'ஆர்', 'S': 'எஸ்', 'T': 'டி', 'U': 'யூ', 'V': 'வி', 'W': 'டபிள்யூ', 'X': 'எக்ஸ்', 'Y': 'வை', 'Z': 'ஜெட்'
    },
    'kn-IN': { 
        '1': 'ಒಂದು', '2': 'ಎರಡು', '3': 'ಮೂರು', '4': 'ನಾಲ್ಕು', '5': 'ಐದು', '6': 'ಆರು', '7': 'ಏಳು', '8': 'ಎಂಟು', '9': 'ಒಂಬತ್ತು',
        'A': 'ಎ', 'B': 'ಬಿ', 'C': 'ಸಿ', 'D': 'ಡಿ', 'E': 'ಇ', 'F': 'ಎಫ್', 'G': 'ಜಿ', 'H': 'ಎಚ್', 'I': 'ಐ', 'J': 'ಜೆ', 'K': 'ಕೆ', 'L': 'ಎಲ್', 'M': 'ಎಂ',
        'N': 'ಎನ್', 'O': 'ಓ', 'P': 'ಪಿ', 'Q': 'ಕ್ಯೂ', 'R': 'ಆರ್', 'S': 'ಎಸ್', 'T': 'ಟಿ', 'U': 'ಯೂ', 'V': 'ವಿ', 'W': 'ಡಬ್ಲ್ಯೂ', 'X': 'ಎಕ್ಸ್', 'Y': 'ವೈ', 'Z': 'ಝೆಡ್'
    },
    'bn-IN': { 
        '1': 'এক', '2': 'দুই', '3': 'তিন', '4': 'চার', '5': 'পাঁচ', '6': 'ছয়', '7': 'সাত', '8': 'আট', '9': 'নয়',
        'A': 'এ', 'B': 'বি', 'C': 'সি', 'D': 'ডি', 'E': 'ই', 'F': 'এফ', 'G': 'জি', 'H': 'এইচ', 'I': 'আই', 'J': 'জে', 'K': 'কে', 'L': 'এল', 'M': 'এম',
        'N': 'এন', 'O': 'ও', 'P': 'পি', 'Q': 'কিউ', 'R': 'আর', 'S': 'এস', 'T': 'টি', 'U': 'ইউ', 'V': 'ভি', 'W': 'ডাবলিউ', 'X': 'এক্স', 'Y': 'ওয়াই', 'Z': 'জেড'
    },
    'mr-IN': { 
        '1': 'एक', '2': 'दोन', '3': 'तीन', '4': 'चार', '5': 'पाच', '6': 'सहा', '7': 'सात', '8': 'आठ', '9': 'नऊ',
        'A': 'ए', 'B': 'बी', 'C': 'सी', 'D': 'डी', 'E': 'ई', 'F': 'एफ', 'G': 'जी', 'H': 'एच', 'I': 'आय', 'J': 'जे', 'K': 'के', 'L': 'एल', 'M': 'एम',
        'N': 'एन', 'O': 'ओ', 'P': 'पी', 'Q': 'क्यू', 'R': 'आर', 'S': 'एस', 'T': 'टी', 'U': 'यू', 'V': 'व्ही', 'W': 'डब्ल्यू', 'X': 'एक्स', 'Y': 'वाय', 'Z': 'झेड'
    }
}

short_lang_mappings = {
    'en': 'en-US', 'hi': 'hi-IN', 'te': 'te-IN', 'ml': 'ml-IN',
    'ta': 'ta-IN', 'kn': 'kn-IN', 'bn': 'bn-IN', 'mr': 'mr-IN'
}

# ===================================================================
# ### THIS IS THE FINAL CORRECTED TTS FUNCTION ###
# ===================================================================
@app.route('/tts', methods=['GET', 'HEAD'])
def text_to_speech():
    # Handle HEAD requests for pre-flight checks
    if request.method == 'HEAD':
        return '', 200
    
    try:
        text_to_speak = request.args.get('text')
        lang = request.args.get('lang', 'en')
        
        if not text_to_speak:
            return jsonify({'success': False, 'error': 'No text provided'}), 400

        final_text = text_to_speak
        
        if text_to_speak.isdigit():
            full_lang_code = short_lang_mappings.get(lang)
            if full_lang_code:
                translated_word = language_mappings.get(full_lang_code, {}).get(text_to_speak)
                if translated_word:
                    final_text = translated_word
        
        print(f"TTS Request: lang={lang}, original_text='{text_to_speak}', final_text='{final_text}'")
        
        # This line is the critical fix: it encodes the text for the URL
        encoded_text = urllib.parse.quote(final_text)
        
        url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl={lang}&client=tw-ob&q={encoded_text}"
        
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'
        }, stream=True)
        
        if response.status_code != 200:
            print(f"Google TTS error: Status code {response.status_code}")
            return f'Google TTS returned status code {response.status_code}', 502
        
        return Response(response.iter_content(chunk_size=1024), 
                      content_type=response.headers.get('content-type'),
                      headers=dict(response.headers))
    
    except Exception as e:
        print(f"TTS Exception: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# New route for testing TTS
@app.route('/test-tts')
def test_tts_page():
    return render_template('test_tts.html')

# New route for testing TTS with all languages
@app.route('/test-tts-languages')
def test_tts_languages_page():
    return send_from_directory('.', 'test_tts_languages.html')

# Simple TTS test page
@app.route('/simple-tts-test')
def simple_tts_test():
    return send_from_directory('.', 'simple_tts_test.html')

# (The rest of your code is correct and remains unchanged)

# Your existing MediaPipe setup and other code...
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Temporarily redirect stderr during model loading
original_stderr = sys.stderr
sys.stderr = DevNull()
model = tf.keras.models.load_model('model.h5')

# Initialize Whisper model lazily (only when needed)
whisper_model = None

sys.stderr = original_stderr


alphabet = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# --- Add landmark processing functions from isl_detection.py ---
def calc_landmark_list(image, landmarks):
    try:
        if image is None or len(image.shape) < 2:
            print("Invalid image format in calc_landmark_list")
            return None
            
        image_width, image_height = image.shape[1], image.shape[0]
        if image_width <= 0 or image_height <= 0:
            print("Invalid image dimensions")
            return None
            
        landmark_point = []
        for _, landmark in enumerate(landmarks.landmark):
            if not hasattr(landmark, 'x') or not hasattr(landmark, 'y'):
                print("Invalid landmark format")
                continue
                
            # Ensure coordinates are within image bounds
            landmark_x = min(max(int(landmark.x * image_width), 0), image_width - 1)
            landmark_y = min(max(int(landmark.y * image_height), 0), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])
            
        if not landmark_point:
            print("No valid landmarks found")
            return None
            
        return landmark_point
    except Exception as e:
        print(f"Error in calc_landmark_list: {str(e)}")
        return None

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

# Updated to use the same pipeline as isl_detection.py
def extract_landmarks(frame):
    try:
        # Ensure frame is not None and has valid dimensions
        if frame is None or len(frame.shape) != 3:
            print("Invalid frame format")
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    # Use the same landmark extraction and preprocessing as original script
                    landmark_list = calc_landmark_list(frame, hand_landmarks)
                    if not landmark_list:
                        print("No landmarks detected")
                        continue
                        
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    if not pre_processed_landmark_list:
                        print("Landmark preprocessing failed")
                        continue
                        
                    return np.array(pre_processed_landmark_list)
                except Exception as e:
                    print(f"Error processing landmarks: {str(e)}")
                    continue
        else:
            print("No hand landmarks detected")
        return None
    except Exception as e:
        print(f"Error in extract_landmarks: {str(e)}")
        return None

@app.route('/test-voice')
def test_voice():
    """Simple voice test page"""
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Microphone Permission Test</title>
</head>
<body>
    <h1>🎤 Microphone Permission Test</h1>
    <button id="testPermBtn">1. Test Microphone Permission</button>
    <button id="recordBtn" disabled>2. Record Audio</button>
    <div id="status" style="margin: 20px 0; padding: 10px; border: 1px solid #ccc;"></div>
    <div id="result"></div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let stream = null;
        
        function updateStatus(message, color = 'black') {
            document.getElementById('status').innerHTML = `<div style="color: ${color};">${message}</div>`;
            console.log(message);
        }
        
        document.getElementById('testPermBtn').addEventListener('click', async () => {
            updateStatus('Testing microphone permissions...', 'blue');
            
            try {
                // Check if browser supports getUserMedia
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    updateStatus('❌ Browser does not support microphone access. Please use Chrome, Firefox, or Edge.', 'red');
                    return;
                }
                
                updateStatus('✅ Browser supports microphone API. Requesting access...', 'green');
                
                // Request microphone access
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 44100,
                        channelCount: 1
                    }
                });
                
                updateStatus('✅ Microphone permission granted! You can now record.', 'green');
                document.getElementById('recordBtn').disabled = false;
                
                // Test MediaRecorder
                if (!window.MediaRecorder) {
                    updateStatus('❌ Browser does not support MediaRecorder API', 'red');
                    return;
                }
                
                updateStatus('✅ All checks passed! Recording should work.', 'green');
                
            } catch (error) {
                console.error('Permission error:', error);
                let message = 'Microphone permission failed: ';
                if (error.name === 'NotAllowedError') {
                    message += 'Permission denied. Please click "Allow" when prompted.';
                } else if (error.name === 'NotFoundError') {
                    message += 'No microphone found. Please connect a microphone.';
                } else if (error.name === 'NotReadableError') {
                    message += 'Microphone is being used by another application.';
                } else {
                    message += error.message;
                }
                updateStatus('❌ ' + message, 'red');
            }
        });
        
        document.getElementById('recordBtn').addEventListener('click', async () => {
            if (!isRecording) {
                try {
                    updateStatus('🎤 Starting recording for 3 seconds...', 'blue');
                    
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                        updateStatus(\`📊 Audio chunk received: \${event.data.size} bytes\`, 'green');
                    };
                    
                    mediaRecorder.onstop = async () => {
                        updateStatus('🔄 Recording stopped, sending to server...', 'blue');
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        
                        updateStatus(\`📦 Created audio blob: \${audioBlob.size} bytes\`, 'green');
                        
                        const formData = new FormData();
                        formData.append('audio', audioBlob, 'recording.wav');
                        formData.append('language', 'en');
                        
                        try {
                            updateStatus('📤 Sending to Whisper API...', 'blue');
                            const response = await fetch('/speech-to-text', {
                                method: 'POST',
                                body: formData
                            });
                            
                            const result = await response.json();
                            console.log('Server response:', result);
                            
                            updateStatus('✅ Response received from server', 'green');
                            document.getElementById('result').innerHTML = 
                                \`<h3>Whisper Result:</h3><pre style="background: #f0f0f0; padding: 10px;">\${JSON.stringify(result, null, 2)}</pre>\`;
                                
                        } catch (error) {
                            console.error('Server error:', error);
                            updateStatus('❌ Server error: ' + error.message, 'red');
                        }
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    document.getElementById('recordBtn').textContent = '🛑 Stop Recording';
                    
                    // Auto-stop after 3 seconds
                    setTimeout(() => {
                        if (isRecording && mediaRecorder.state === 'recording') {
                            mediaRecorder.stop();
                            isRecording = false;
                            document.getElementById('recordBtn').textContent = '2. Record Audio';
                        }
                    }, 3000);
                    
                } catch (error) {
                    updateStatus('❌ Recording error: ' + error.message, 'red');
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                document.getElementById('recordBtn').textContent = '2. Record Audio';
            }
        });
        
        // Initial status
        updateStatus('Click "Test Microphone Permission" first to check if your microphone works.');
    </script>
</body>
</html>'''

@app.route('/')
def index():
    response = make_response(render_template('index.html'))
    # Disable caching for HTML to prevent old JavaScript from being served
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if data is None or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Invalid request: no image data'
            })
        
        # Get language from request or default to English
        language = data.get('language', 'en-US')
        
        image_data = request.json['image']
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        # Convert PIL Image to numpy array and ensure correct color format
        frame = np.array(image)
        if len(frame.shape) == 2:  # If grayscale, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # If RGBA, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        # Convert to BGR for OpenCV processing
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print("Frame shape:", frame.shape)  # Debug print
        landmarks = extract_landmarks(frame)
        
        if landmarks is not None:
            # Already preprocessed and 1D
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction, axis=1)
            confidence = float(np.max(prediction))
            predicted_sign = alphabet[predicted_class[0]]
            
            # Get the localized text for the sign (both numbers and alphabets)
            localized_text = predicted_sign
            try:
                # Try to get localized text for both numbers and alphabets
                print(f"Attempting localization: sign={predicted_sign}, language={language}")
                if language in language_mappings and predicted_sign in language_mappings[language]:
                    localized_text = language_mappings[language][predicted_sign]
                    print(f"Direct match found: {localized_text}")
                # Try with short language code if the full code didn't match
                elif language.split('-')[0] in short_lang_mappings:
                    short_lang = language.split('-')[0]
                    full_lang = short_lang_mappings[short_lang]
                    print(f"Trying with short lang: {short_lang} -> {full_lang}")
                    if full_lang in language_mappings and predicted_sign in language_mappings[full_lang]:
                        localized_text = language_mappings[full_lang][predicted_sign]
                        print(f"Short lang match found: {localized_text}")
            except Exception as e:
                # If anything goes wrong with localization, just use the sign itself
                print(f"Localization error: {str(e)}")
                localized_text = predicted_sign
            
            print(f"Final result: sign={predicted_sign}, localized={localized_text}, language={language}")
            return jsonify({
                'success': True,
                'sign': predicted_sign,
                'localized_text': localized_text,
                'confidence': confidence
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No hand detected'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def extract_sign_from_composite(character):
    """Extract individual sign from allGestures.png based on character"""
    try:
        composite_img = Image.open('images/allGestures.png')
        img_width, img_height = composite_img.size
        
        # Define layout - ADJUST THIS based on your actual image arrangement
        chars_layout = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
            ['J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '']
        ]
        
        # Find character position
        char_row, char_col = None, None
        for row_idx, row in enumerate(chars_layout):
            if character in row:
                char_row = row_idx
                char_col = row.index(character)
                break
        
        if char_row is None or char_col is None:
            return None
        
        # Calculate cell dimensions
        cols = len(chars_layout[0])
        rows = len(chars_layout)
        cell_width = img_width // cols
        cell_height = img_height // rows
        
        # Extract with padding
        padding = 10
        left = max(0, char_col * cell_width - padding)
        top = max(0, char_row * cell_height - padding)
        right = min(img_width, (char_col + 1) * cell_width + padding)
        bottom = min(img_height, (char_row + 1) * cell_height + padding)
        
        # Crop and resize
        sign_img = composite_img.crop((left, top, right, bottom))
        sign_img = sign_img.resize((200, 200), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        sign_img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
        
    except Exception as e:
        print(f"Error extracting sign for {character}: {str(e)}")
        return None


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """Convert uploaded audio to text using Whisper - optimized for A-Z and 1-9"""
    try:
        print("Received speech-to-text request")
        
        # Load Whisper model if not already loaded
        global whisper_model
        if whisper_model is None:
            print("Loading Whisper tiny model for A-Z and 1-9 recognition...")
            import whisper
            # Use tiny model for faster loading and processing (39MB)
            whisper_model = whisper.load_model("tiny")
            print("Whisper tiny model loaded successfully!")
        
        # Check if audio file is in request
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No audio file selected'
            }), 400
        
        # Get language parameter (optional)
        language = request.form.get('language', None)
        
        # Save uploaded file temporarily with proper file handling
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
        try:
            # Write audio data to temporary file
            with os.fdopen(temp_fd, 'wb') as temp_file:
                audio_file.seek(0)  # Reset file pointer
                temp_file.write(audio_file.read())
            
            print(f"Audio saved to temporary file: {temp_path}")
            
            # Verify file exists and has content
            if not os.path.exists(temp_path):
                raise FileNotFoundError(f"Temporary file {temp_path} was not created")
            
            file_size = os.path.getsize(temp_path)
            print(f"Temporary file size: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("Audio file is empty")
            
            print(f"Processing audio file: {temp_path} for language: {language}")
            
            # Use Whisper to transcribe the audio with speed optimizations
            print("Starting fast Whisper transcription...")
            
            # Optimize Whisper settings for speed - critical for user experience
            options = {
                "fp16": False,  # Use float32 for CPU compatibility and speed
                "no_speech_threshold": 0.1,  # Lower threshold for short recordings
                "compression_ratio_threshold": 2.4,  # Less strict compression check
                "temperature": 0.0,  # Deterministic output - faster
                "best_of": 1,  # Don't generate multiple candidates - much faster
                "beam_size": 1,  # Fastest beam search
                "word_timestamps": False,  # Skip word timing calculation
                "condition_on_previous_text": False,  # Don't use context - faster
            }
            
            # For numbers 1-9, we want to detect in the original language
            # For alphabets A-Z, they're the same across languages
            if language and language != 'auto':
                # Convert language codes if needed (hi-IN -> hi)
                whisper_lang = language.split('-')[0] if '-' in language else language
                result = whisper_model.transcribe(temp_path, language=whisper_lang, **options)
            else:
                # Auto-detect language with optimized settings
                result = whisper_model.transcribe(temp_path, **options)
            
            transcribed_text = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            print(f"Transcription result: '{transcribed_text}' (detected language: {detected_language})")
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if not transcribed_text:
                return jsonify({
                    'success': False,
                    'error': 'No speech detected in audio'
                })
            
            # Extract relevant character/number from transcribed text
            extracted_char = extract_character_from_speech(transcribed_text, detected_language)
            
            return jsonify({
                'success': True,
                'text': extracted_char if extracted_char else transcribed_text,
                'detected_language': detected_language,
                'original_text': transcribed_text,
                'extracted_character': extracted_char
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise e
            
    except Exception as e:
        print(f"Error in speech_to_text: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Speech recognition error: {str(e)}'
        }), 500


def extract_character_from_speech(text, detected_language):
    """Extract A-Z or 1-9 from speech text based on detected language"""
    try:
        clean_text = text.lower().strip()
        print(f"Extracting character from: '{clean_text}' (language: {detected_language})")
        
        # First check for single alphabets (same across all languages)
        alphabet_match = None
        for char in 'abcdefghijklmnopqrstuvwxyz':
            if char in clean_text.split():  # Check if it's a separate word
                alphabet_match = char.upper()
                break
            # Also check if the whole text is just the letter
            if clean_text == char:
                alphabet_match = char.upper()
                break
        
        if alphabet_match:
            print(f"Found alphabet: {alphabet_match}")
            return alphabet_match
        
        # Check for numbers 1-9 in different languages
        number_mappings = {
            # English
            'en': {
                'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
            },
            # Hindi numbers
            'hi': {
                'एक': '1', 'दो': '2', 'तीन': '3', 'चार': '4', 'पांच': '5', 'पाँच': '5',
                'छह': '6', 'छः': '6', 'सात': '7', 'आठ': '8', 'नौ': '9'
            },
            # Telugu numbers  
            'te': {
                'ఒకటి': '1', 'రెండు': '2', 'మూడు': '3', 'నాలుగు': '4', 'ఐదు': '5',
                'ఆరు': '6', 'ఏడు': '7', 'ఎనిమిది': '8', 'తొమ్మిది': '9'
            },
            # Malayalam numbers
            'ml': {
                'ഒന്ന്': '1', 'രണ്ട്': '2', 'മൂന്ന്': '3', 'നാല്': '4', 'അഞ്ച്': '5',
                'ആറ്': '6', 'ഏഴ്': '7', 'എട്ട്': '8', 'ഒമ്പത്': '9'
            },
            # Tamil numbers
            'ta': {
                'ஒன்று': '1', 'இரண்டு': '2', 'மூன்று': '3', 'நான்கு': '4', 'ஐந்து': '5',
                'ஆறு': '6', 'ஏழு': '7', 'எட்டு': '8', 'ஒன்பது': '9'
            },
            # Kannada numbers
            'kn': {
                'ಒಂದು': '1', 'ಎರಡು': '2', 'ಮೂರು': '3', 'ನಾಲ್ಕು': '4', 'ಐದು': '5',
                'ಆರು': '6', 'ಏಳು': '7', 'ಎಂಟು': '8', 'ಒಂಬತ್ತು': '9'
            },
            # Bengali numbers
            'bn': {
                'এক': '1', 'দুই': '2', 'তিন': '3', 'চার': '4', 'পাঁচ': '5',
                'ছয়': '6', 'সাত': '7', 'আট': '8', 'নয়': '9'
            },
            # Marathi numbers
            'mr': {
                'एक': '1', 'दोन': '2', 'तीन': '3', 'चार': '4', 'पाच': '5',
                'सहा': '6', 'सात': '7', 'आठ': '8', 'नऊ': '9'
            }
        }
        
        # Check in detected language first
        lang_mappings = number_mappings.get(detected_language, {})
        for word, number in lang_mappings.items():
            if word in text or word.lower() in clean_text:
                print(f"Found number in {detected_language}: {word} -> {number}")
                return number
        
        # If not found in detected language, try all languages
        for lang, mappings in number_mappings.items():
            if lang != detected_language:  # Skip already checked language
                for word, number in mappings.items():
                    if word in text or word.lower() in clean_text:
                        print(f"Found number in {lang}: {word} -> {number}")
                        return number
        
        # Check for direct digit in text
        digit_match = None
        for digit in '123456789':
            if digit in clean_text:
                digit_match = digit
                break
        
        if digit_match:
            print(f"Found digit: {digit_match}")
            return digit_match
        
        # If nothing found, try to extract first meaningful word that could be a letter
        words = clean_text.split()
        for word in words:
            if len(word) == 1 and word.isalpha():
                print(f"Found single letter: {word.upper()}")
                return word.upper()
        
        print("No valid character found")
        return None
        
    except Exception as e:
        print(f"Error extracting character: {str(e)}")
        return None


@app.route('/text-to-sign', methods=['POST'])
def text_to_sign():
    try:
        data = request.json
        input_text = data.get('text', '').strip()
        language = data.get('language', 'en-US')
        
        print(f"Received text-to-sign request: text='{input_text}', language='{language}'")
        
        def find_character_in_language(text, lang_code):
            """Enhanced function to find character in language mappings"""
            if lang_code not in language_mappings:
                return None
                
            forward_map = language_mappings[lang_code]
            reverse_map = {}
            
            # Build comprehensive reverse mapping with multiple variations
            for eng_char, local_text in forward_map.items():
                # Add exact match
                reverse_map[local_text] = eng_char
                # Add lowercase match
                reverse_map[local_text.lower()] = eng_char
                # Add uppercase match
                reverse_map[local_text.upper()] = eng_char
                # Add alphanumeric only version
                clean_text = ''.join(c for c in local_text if c.isalnum())
                if clean_text:
                    reverse_map[clean_text] = eng_char
                    reverse_map[clean_text.lower()] = eng_char
                    reverse_map[clean_text.upper()] = eng_char
            
            # Try exact matches first
            if text in reverse_map:
                return reverse_map[text]
            if text.lower() in reverse_map:
                return reverse_map[text.lower()]
            if text.upper() in reverse_map:
                return reverse_map[text.upper()]
                
            # Try alphanumeric only version
            clean_input = ''.join(c for c in text if c.isalnum())
            if clean_input and clean_input in reverse_map:
                return reverse_map[clean_input]
                
            return None
        
        def is_english_character(text):
            """Check if input is already an English character"""
            if len(text) == 1:
                return text.upper() in alphabet
            return False
        
        target_char = None
        found_language = None
        
        # If it's already an English character, use it directly
        if is_english_character(input_text):
            target_char = input_text.upper()
            found_language = 'en-US'
            print(f"Direct English character: {input_text} -> {target_char}")
        else:
            # Try to find the character in all supported languages
            # Start with the specified language
            target_char = find_character_in_language(input_text, language)
            if target_char:
                found_language = language
                print(f"Found in specified language {language}: {input_text} -> {target_char}")
            
            # If not found and language has country code, try base language
            if not target_char and '-' in language:
                base_lang = language.split('-')[0]
                if base_lang in short_lang_mappings:
                    full_lang = short_lang_mappings[base_lang]
                    if full_lang != language:  # Only try if different
                        target_char = find_character_in_language(input_text, full_lang)
                        if target_char:
                            found_language = full_lang
                            print(f"Found in base language {full_lang}: {input_text} -> {target_char}")
            
            # If still not found, try all other supported languages
            if not target_char:
                for lang_code in language_mappings.keys():
                    if lang_code != language and lang_code != found_language:
                        target_char = find_character_in_language(input_text, lang_code)
                        if target_char:
                            found_language = lang_code
                            print(f"Found in language {lang_code}: {input_text} -> {target_char}")
                            break
        
        # If we found a valid character, return the sign
        if target_char and target_char in alphabet:
            # Get the localized text for display in the requested language
            displayed_text = input_text
            if language in language_mappings and target_char in language_mappings[language]:
                displayed_text = language_mappings[language][target_char]
            elif found_language and found_language in language_mappings:
                displayed_text = language_mappings[found_language][target_char]
            
            # Check if sign image exists
            static_url = f"/images/signs/{target_char}.png"
            static_path = os.path.join(app.root_path, 'images', 'signs', f'{target_char}.png')
            
            if os.path.exists(static_path):
                return jsonify({
                    'success': True,
                    'character': target_char,
                    'sign_image': static_url,
                    'input_text': displayed_text,
                    'original_input': input_text,
                    'detected_language': found_language or language
                })
            else:
                return jsonify({
                    'success': False,
                    'error': f'Sign image for "{target_char}" not found in images/signs/ folder'
                })
        else:
            # Provide helpful error message with examples in the user's language
            if language in language_mappings:
                # Get examples from the specified language
                examples = []
                lang_map = language_mappings[language]
                if '1' in lang_map:
                    examples.append(f"'{lang_map['1']}' (1)")
                if 'A' in lang_map:
                    examples.append(f"'{lang_map['A']}' (A)")
                
                example_text = ", ".join(examples) if examples else "1, A"
                error_msg = f'Character "{input_text}" not recognized. Try numbers (1-9) or letters (A-Z). Examples: {example_text}'
            else:
                error_msg = f'Character "{input_text}" not supported. Use numbers (1-9) or letters (A-Z)'
                
            return jsonify({
                'success': False,
                'error': error_msg,
                'supported_languages': list(language_mappings.keys())
            })
            
    except Exception as e:
        print(f"Error in text_to_sign: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Test routes for debugging
@app.route('/analyze-image')
def analyze_image():
    try:
        img = Image.open('images/allGestures.png')
        width, height = img.size
        return jsonify({
            'width': width,
            'height': height,
            'suggested_cell_width': width // 9,
            'suggested_cell_height': height // 4
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/test-extract/<character>')
def test_extract(character):
    try:
        sign_data = extract_sign_from_composite(character.upper())
        if sign_data:
            return f'<img src="{sign_data}" style="max-width:300px;">'
        else:
            return f'Could not extract sign for {character}'
    except Exception as e:
        return f'Error: {str(e)}'

# Custom Flask run function to show only the URL
def run_flask_app():
    import threading
    import time
    
    def clear_and_show_url():
        time.sleep(2)  # Give Flask a moment to start
        # Clear console (works on Windows)
        os.system('cls' if os.name == 'nt' else 'clear')
        print("\n" + "="*60)
        print("   ISL DETECTION APP RUNNING")
        print("   Open in your browser: http://localhost:5000")
        print("   Press CTRL+C to stop the server")
        print("="*60 + "\n")
    
    # Start the URL display in a separate thread
    url_thread = threading.Thread(target=clear_and_show_url)
    url_thread.daemon = True
    url_thread.start()
    
    # Start Flask with minimal logging
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # Disable Flask's default startup messages
    cli = sys.modules['flask.cli']
    cli.show_server_banner = lambda *x: None
    
    # Configure logging
    log = logging.getLogger('werkzeug')
    log.disabled = True
    
    # Start the app with custom runner
    run_flask_app()