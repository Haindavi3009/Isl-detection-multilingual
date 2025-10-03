#!/usr/bin/env python3
"""
Simple test for text-to-sign API using direct HTTP requests
"""
import requests
import json

def test_single_request(text, language, expected_char):
    """Test a single text-to-sign request"""
    url = "http://localhost:5000/text-to-sign"
    payload = {"text": text, "language": language}
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                actual_char = result.get('character')
                sign_image = result.get('sign_image', 'No image')
                
                status = "PASS" if actual_char == expected_char else "FAIL"
                print(f"{status}: {language} '{text}' -> {actual_char} (expected {expected_char})")
                print(f"     Sign image: {sign_image}")
                return actual_char == expected_char
            else:
                print(f"FAIL: {language} '{text}' -> Error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"FAIL: {language} '{text}' -> HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"FAIL: {language} '{text}' -> Connection error: {str(e)}")
        return False

def main():
    print("Testing text-to-sign functionality...")
    print("=" * 50)
    
    # Test a few key cases
    test_cases = [
        ("A", "en-US", "A"),
        ("1", "en-US", "1"),
        ("एक", "hi-IN", "1"),  # Hindi "one"
        ("ए", "hi-IN", "A"),   # Hindi "A"
    ]
    
    passed = 0
    total = len(test_cases)
    
    for text, language, expected in test_cases:
        if test_single_request(text, language, expected):
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
    else:
        print("⚠️ Some tests failed.")

if __name__ == "__main__":
    main()