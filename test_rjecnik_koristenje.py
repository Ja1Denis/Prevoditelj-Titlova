import re
from improved_translator import ImprovedSubtitleTranslator

# Kreiraj instancu prevoditelja s vašim rječnikom
translator = ImprovedSubtitleTranslator(user_dict_path="osnovni_rijecnik.txt")

# Testni primjeri
testni_tekstovi = [
    "Well, I don't know about that.",
    "So, what do you think?",
    "There you go, that's what I meant.",
    "The meaning of life is...",
    "Oh come on, don't be like that!"
]

# Testiranje prijevoda
print("🔍 Testiranje korištenja rječnika:\n" + "="*60)

for tekst in testni_tekstovi:
    print(f"\n📝 Original: {tekst}")
    
    # Primijeni samo korisnički rječnik
    prevedeno = translator._apply_false_friends(tekst, apply_discourse_markers=True)
    
    if prevedeno != tekst:
        print(f"✅ Promijenjeno u: {prevedeno}")
    else:
        print("❌ Nema promjena (nema podudaranja u rječniku)")

# Dodatni test s rečenicom
recenica = "Well, so there you go, that's the meaning of life."
print("\n🔍 Testiranje rečenice:" + "="*50)
print(f"Original: {recenica}")
prevedeno = translator._apply_false_friends(recenica)
print(f"Nakon rječnika: {prevedeno}")

# Ispis broja učitranih pravila iz rječnika
try:
    print(f"\n📊 Ukupno učitano {len(translator._user_pairs)} pravila iz rječnika")
    if hasattr(translator, '_user_pairs') and translator._user_pairs:
        print("Primjeri učitranih pravila:")
        for i, (pattern, replacement, priority) in enumerate(translator._user_pairs[:5], 1):
            print(f"  {i}. {pattern.pattern} -> {replacement} (prioritet: {priority})")
except Exception as e:
    print(f"\n⚠️ Greška pri dohvaćanju informacija o rječniku: {e}")
    
# Testiranje s nekim specifičnim primjerima
print("\n🔍 Testiranje specifičnih primjera:" + "="*50)
test_cases = [
    "Well, that's interesting.",
    "So, let's begin.",
    "There you go, all done!",
    "The meaning is clear.",
    "Oh come on, don't be silly!"
]

for test in test_cases:
    result = translator._apply_false_friends(test)
    print(f"\n'{test}'")
    print(f"  ⮕ '{result}'" + ("" if test == result else "  ✅"))
