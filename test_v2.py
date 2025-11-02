from improved_translator import ImprovedSubtitleTranslator
import re

def test_dictionary(dict_path):
    # Initialize translator with the test dictionary
    translator = ImprovedSubtitleTranslator(
        model_name="Helsinki-NLP/opus-mt-en-zls",
        user_dict_path=dict_path
    )
    
    # Test individual words
    test_cases = [
        # Basic words
        ("well", "pa"),
        ("so", "dakle"),
        ("hey", "hej"),
        ("oh", "o"),
        ("look", "gledaj"),
        ("listen", "slušaj"),
        ("alright", "u redu"),
        ("okay", "u redu"),
        ("oh my god", "o moj bože"),
        ("come on", "hajde"),
        
        # Words with punctuation
        ("well,", "pa,"),
        ("so?", "dakle?"),
        ("hey!", "hej!"),
        ("oh!", "o!"),
        ("look!", "gledaj!"),
        ("listen,", "slušaj,"),
        ("oh my god!", "o moj bože!"),
        ("come on!", "hajde!")
    ]
    
    print("Testing individual words...\n" + "="*50)
    
    # Test each word
    for original, expected in test_cases:
        translated = translator._apply_false_friends(original)
        status = "✅" if translated.lower() == expected.lower() else "❌"
        print(f"{status} {original:<15} -> {translated:<15} (expected: {expected})")
    
    # Test full sentences
    print("\nTesting full sentences...\n" + "="*50)
    
    sentences = [
        "Well, I don't know",
        "So what do you think?",
        "Hey, how are you?",
        "Oh, I see what you mean",
        "Look at that beautiful view",
        "Listen to this song",
        "Alright, I'll do it",
        "Okay, let's go",
        "Oh my god, that's amazing!",
        "Come on, you can do it!"
    ]
    
    for sentence in sentences:
        translated = translator._apply_false_friends(sentence)
        print(f"Original: {sentence}")
        print(f"Translated: {translated}")
        print("-" * 60)

if __name__ == "__main__":
    dict_path = "test_dict_v2.txt"  # Using our improved test dictionary
    print(f"Testing with dictionary: {dict_path}\n")
    test_dictionary(dict_path)
