#!/usr/bin/env python3
"""
Test script for text-to-sign functionality across all supported languages
"""
import requests
import json

# Test server URL
URL = "http://localhost:5000/text-to-sign"

# Test cases for all 8 supported languages
test_cases = [
    # English
    {"text": "A", "language": "en-US", "expected": "A"},
    {"text": "1", "language": "en-US", "expected": "1"},
    
    # Telugu
    {"text": "ఎ", "language": "te-IN", "expected": "A"},
    {"text": "ఒకటి", "language": "te-IN", "expected": "1"},
    
    # Hindi  
    {"text": "ए", "language": "hi-IN", "expected": "A"},
    {"text": "एक", "language": "hi-IN", "expected": "1"},
    
    # Malayalam
    {"text": "എ", "language": "ml-IN", "expected": "A"},
    {"text": "ഒന്ന്", "language": "ml-IN", "expected": "1"},
    
    # Tamil
    {"text": "ஏ", "language": "ta-IN", "expected": "A"},
    {"text": "ஒன்று", "language": "ta-IN", "expected": "1"},
    
    # Kannada
    {"text": "ಎ", "language": "kn-IN", "expected": "A"},
    {"text": "ಒಂದು", "language": "kn-IN", "expected": "1"},
    
    # Bengali
    {"text": "এ", "language": "bn-IN", "expected": "A"},
    {"text": "এক", "language": "bn-IN", "expected": "1"},
    
    # Marathi
    {"text": "ए", "language": "mr-IN", "expected": "A"},
    {"text": "एक", "language": "mr-IN", "expected": "1"},
]

def test_text_to_sign():
    print("Testing text-to-sign functionality for all 8 languages...")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            response = requests.post(URL, json=test_case, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    actual_char = result.get('character')
                    if actual_char == test_case['expected']:
                        print(f"✓ Test {i:2d}: PASS - {test_case['language']} '{test_case['text']}' -> {actual_char}")
                        passed += 1
                    else:
                        print(f"✗ Test {i:2d}: FAIL - {test_case['language']} '{test_case['text']}' -> expected {test_case['expected']}, got {actual_char}")
                        failed += 1
                else:
                    print(f"✗ Test {i:2d}: FAIL - {test_case['language']} '{test_case['text']}' -> {result.get('error', 'Unknown error')}")
                    failed += 1
            else:
                print(f"✗ Test {i:2d}: FAIL - HTTP {response.status_code}")
                failed += 1
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Test {i:2d}: FAIL - Connection error: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"✗ Test {i:2d}: FAIL - Error: {str(e)}")
            failed += 1
    
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    if failed == 0:
        print("🎉 All tests passed! Text-to-sign works for all 8 languages!")
    else:
        print(f"⚠️  {failed} tests failed. Check the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    test_text_to_sign()