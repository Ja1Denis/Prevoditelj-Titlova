from improved_translator import ImprovedSubtitleTranslator

def test_dictionary(dict_path):
    # Initialize translator with your dictionary
    translator = ImprovedSubtitleTranslator(
        model_name="Helsinki-NLP/opus-mt-en-zls",
        user_dict_path=dict_path
    )
    
    # Test phrases with expected translations
    test_cases = [
        ("well, I don't know", "pa, I don't know"),
        ("so what?", "pa what?"),
        ("hey there", "hej there"),
        ("well then let's go", "e pa let's go"),
        ("meaning of life", "znači of life"),
        ("oh come on, really?", "ma daj, really?"),
        ("oh my god!", "o my god!"),
        ("alright, I'll do it", "u redu, I'll do it"),
        ("look at that", "vidi at that"),
        ("listen to me", "slušaj to me")
    ]
    
    print("Testing dictionary translations...\n" + "="*50)
    
    # First, test with context
    print("\nTesting with context...")
    for original, expected in test_cases:
        translated = translator._apply_false_friends(original)
        status = "✅" if translated.lower() == expected.lower() else "❌"
        print(f"{status} Original: {original}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {translated}")
        print("-" * 60)
    
    # Then test without context
    print("\nTesting without context...")
    for original, _ in test_cases:
        translated = translator._apply_false_friends(original, apply_discourse_markers=False)
        print(f"Original: {original}")
        print(f"Translated: {translated}")
        print("-" * 60)

if __name__ == "__main__":
    dict_path = r"F:\\Serije\\osnovni_riječnik.txt"
    print(f"Testing dictionary file: {dict_path}")
    test_dictionary(dict_path)
