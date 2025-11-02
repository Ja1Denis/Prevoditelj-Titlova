from improved_translator import ImprovedSubtitleTranslator
import re

def test_dictionary(dict_path):
    # Initialize translator with your dictionary
    translator = ImprovedSubtitleTranslator(
        model_name="Helsinki-NLP/opus-mt-en-zls",
        user_dict_path=dict_path
    )
    
    # Test the dictionary directly
    print("Testing dictionary entries...\n" + "="*50)
    
    # Simple test - check if any entries were loaded
    if not hasattr(translator, '_user_pairs') or not translator._user_pairs:
        print("❌ No dictionary entries were loaded!")
        return
        
    print(f"✅ Loaded {len(translator._user_pairs)} dictionary entries")
    
    # Print first 10 entries for verification
    print("\nFirst 10 dictionary entries:")
    for i, (pattern, repl, prio) in enumerate(translator._user_pairs[:10]):
        print(f"{i+1}. Pattern: {pattern.pattern} -> {repl} (priority: {prio})")
    
    # Test specific translations
    test_cases = [
        "well",
        "so",
        "hey",
        "oh",
        "look",
        "listen"
    ]
    
    print("\nTesting specific translations:")
    for word in test_cases:
        translated = translator._apply_false_friends(word)
        print(f"{word} -> {translated}")

if __name__ == "__main__":
    dict_path = r"F:\Serije\osnovni_riječnik.txt"
    print(f"Testing dictionary file: {dict_path}")
    test_dictionary(dict_path)
