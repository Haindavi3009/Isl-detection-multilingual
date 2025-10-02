import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from flask import Flask, render_template, request, jsonify, Response
import base64
from PIL import Image
import io
import copy
import itertools
import logging
import sys
import requests
import urllib.parse # <-- IMPORT ADDED HERE

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

# Configure Flask logger to only show errors
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# You have correctly moved the dictionaries here. This is perfect.
language_mappings = {
    'en-US': { '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine' },
    'te-IN': { '1': 'ఒకటి', '2': 'రెండు', '3': 'మూడు', '4': 'నాలుగు', '5': 'ఐదు', '6': 'ఆరు', '7': 'ఏడు', '8': 'ఎనిమిది', '9': 'తొమ్మిది' },
    'hi-IN': { '1': 'एक', '2': 'दो', '3': 'तीन', '4': 'चार', '5': 'पांच', '6': 'छह', '7': 'सात', '8': 'आठ', '9': 'नौ' },
    'ml-IN': { '1': 'ഒന്ന്', '2': 'രണ്ട്', '3': 'മൂന്ന്', '4': 'നാല്', '5': 'അഞ്ച്', '6': 'ആറ്', '7': 'ഏഴ്', '8': 'എട്ട്', '9': 'ഒമ്പത്' },
    'ta-IN': { '1': 'ஒன்று', '2': 'இரண்டு', '3': 'மூன்று', '4': 'நான்கு', '5': 'ஐந்து', '6': 'ஆறு', '7': 'ஏழு', '8': 'எட்டு', '9': 'ஒன்பது' },
    'kn-IN': { '1': 'ಒಂದು', '2': 'ಎರಡು', '3': 'ಮೂರು', '4': 'ನಾಲ್ಕು', '5': 'ಐದು', '6': 'ಆರು', '7': 'ಏಳು', '8': 'ಎಂಟು', '9': 'ಒಂಬತ್ತು' },
    'bn-IN': { '1': 'এক', '2': 'দুই', '3': 'তিন', '4': 'চার', '5': 'পাঁচ', '6': 'ছয়', '7': 'সাত', '8': 'আট', '9': 'নয়' },
    'mr-IN': { '1': 'एक', '2': 'दोन', '3': 'तीन', '4': 'चार', '5': 'पाच', '6': 'सहा', '7': 'सात', '8': 'आठ', '9': 'नऊ' }
}

short_lang_mappings = {
    'en': 'en-US', 'hi': 'hi-IN', 'te': 'te-IN', 'ml': 'ml-IN',
    'ta': 'ta-IN', 'kn': 'kn-IN', 'bn': 'bn-IN', 'mr': 'mr-IN'
}

# ===================================================================
# ### THIS IS THE FINAL CORRECTED TTS FUNCTION ###
# ===================================================================
@app.route('/tts', methods=['GET'])
def text_to_speech():
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
model = tf.keras.models.load_model('D:/isl-appp/Indian-Sign-Language-Detection/model.h5')
sys.stderr = original_stderr


alphabet = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# --- Add landmark processing functions from isl_detection.py ---
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Use the same landmark extraction and preprocessing as original script
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            return np.array(pre_processed_landmark_list)
    return None

@app.route('/')
def index():
    return render_template('index.html')

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
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        landmarks = extract_landmarks(frame)
        
        if landmarks is not None:
            # Already preprocessed and 1D
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction, axis=1)
            confidence = float(np.max(prediction))
            predicted_sign = alphabet[predicted_class[0]]
            
            # Get the localized text for the sign
            localized_text = predicted_sign
            try:
                # Only try to get localized text if it's a number (1-9)
                if predicted_sign.isdigit():
                    if language in language_mappings and predicted_sign in language_mappings[language]:
                        localized_text = language_mappings[language][predicted_sign]
                    # Try with short language code if the full code didn't match
                    elif language.split('-')[0] in short_lang_mappings:
                        short_lang = language.split('-')[0]
                        full_lang = short_lang_mappings[short_lang]
                        if full_lang in language_mappings and predicted_sign in language_mappings[full_lang]:
                            localized_text = language_mappings[full_lang][predicted_sign]
            except Exception as e:
                # If anything goes wrong with localization, just use the sign itself
                print(f"Localization error: {str(e)}")
                localized_text = predicted_sign
            
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