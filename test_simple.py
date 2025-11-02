from improved_translator import ImprovedSubtitleTranslator

def test_dictionary(dict_path):
    # Initialize translator with the test dictionary
    translator = ImprovedSubtitleTranslator(
        model_name="Helsinki-NLP/opus-mt-en-zls",
        user_dict_path=dict_path
    )
    
    # Test phrases with expected translations
    test_cases = [
        ("well", "pa"),
        ("so", "dakle"),
        ("hey", "hej"),
        ("oh", "o"),
        ("look", "gledaj"),
        ("listen", "slušaj"),
        ("alright", "u redu"),
        ("okay", "u redu"),
        ("oh my god", "o moj bože"),
        ("come on", "hajde")
    ]
    
    print("Testing dictionary translations...\n" + "="*50)
    
    # Test each phrase
    for original, expected in test_cases:
        # Test at the beginning of a sentence
        test_phrase = f"{original} test"
        translated = translator._apply_false_friends(test_phrase)
        
        # Check if the translation contains the expected word
        status = "✅" if expected.lower() in translated.lower() else "❌"
        
        print(f"{status} Original: {test_phrase}")
        print(f"   Expected to contain: {expected}")
        print(f"   Got: {translated}")
        print("-" * 60)
    
    # Test full sentences
    print("\nTesting full sentences...")
    sentences = [
        "Well, I don't know",
        "So what do you think?",
        "Hey, how are you?",
        "Oh, I see what you mean",
        "Look at that beautiful view",
        "Listen to this song",
        "Alright, I'll do it",
        "Okay, let's go"
    ]
    
    for sentence in sentences:
        translated = translator._apply_false_friends(sentence)
        print(f"Original: {sentence}")
        print(f"Translated: {translated}")
        print("-" * 60)

if __name__ == "__main__":
    dict_path = "test_dict.txt"  # Using our simplified test dictionary
    print(f"Testing dictionary file: {dict_path}")
    test_dictionary(dict_path)
