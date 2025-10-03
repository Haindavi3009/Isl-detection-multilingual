#!/usr/bin/env python3
"""
Test script to verify the optimized A-Z and 1-9 speech recognition
"""

def test_character_extraction():
    """Test the character extraction logic for A-Z and 1-9"""
    
    # Import the function from app.py
    import sys
    sys.path.append('.')
    from app import extract_character_from_speech
    
    # Test cases for different languages
    test_cases = [
        # English
        ("one", "en", "1"),
        ("two", "en", "2"), 
        ("three", "en", "3"),
        ("a", "en", "A"),
        ("bee", "en", "B"),
        
        # Hindi
        ("एक", "hi", "1"),
        ("दो", "hi", "2"),
        ("तीन", "hi", "3"),
        
        # Telugu  
        ("ఒకటి", "te", "1"),
        ("రెండు", "te", "2"),
        ("మూడు", "te", "3"),
        
        # Malayalam
        ("ഒന്ന്", "ml", "1"),
        ("രണ്ട്", "ml", "2"),
        ("മൂന്ന്", "ml", "3"),
        
        # Tamil
        ("ஒன்று", "ta", "1"),
        ("இரண்டு", "ta", "2"),
        ("மூன்று", "ta", "3"),
        
        # Kannada
        ("ಒಂದು", "kn", "1"),
        ("ಎರಡು", "kn", "2"),
        ("ಮೂರು", "kn", "3"),
        
        # Bengali
        ("এক", "bn", "1"),
        ("দুই", "bn", "2"),
        ("তিন", "bn", "3"),
        
        # Marathi
        ("एक", "mr", "1"),
        ("दोन", "mr", "2"),
        ("तीन", "mr", "3"),
    ]
    
    print("Testing optimized A-Z and 1-9 character extraction...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for text, language, expected in test_cases:
        try:
            result = extract_character_from_speech(text, language)
            
            if result == expected:
                print(f"✓ PASS: '{text}' ({language}) -> {result}")
                passed += 1
            else:
                print(f"✗ FAIL: '{text}' ({language}) -> expected {expected}, got {result}")
                failed += 1
                
        except Exception as e:
            print(f"✗ ERROR: '{text}' ({language}) -> {str(e)}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    if failed == 0:
        print("🎉 All tests passed! Optimized speech recognition is working correctly!")
    else:
        print(f"⚠️ {failed} tests failed. Check the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    test_character_extraction()