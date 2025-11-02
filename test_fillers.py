import unittest
from improved_translator import ImprovedSubtitleTranslator
import os

class TestFillers(unittest.TestCase):    
    def setUp(self):
        """Postavljanje testnog okruženja."""
        self.translator = ImprovedSubtitleTranslator()
        self.test_file = "test_fill.srt"
        
    def test_fillers_translation(self):
        """Testira prevođenje ispunjivača u SRT datoteci."""
        # Pročitaj testnu datoteku
        with open(self.test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Podijeli na blokove (svaki blok je jedan titl)
        blocks = content.strip().split('\n\n')
        
        # Očekivani rezultati za svaki blok
        expected_results = [
            "Da.",           # Yes.
            "Pa...",         # Well...
            "Da, znam",      # Yes, I know
            "Pa, mislim..."  # Well, I think...
        ]
        
        # Obradi svaki blok
        for i, block in enumerate(blocks):
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if len(lines) >= 3:  # Ako ima dovoljno linija
                original_text = ' '.join(lines[2:])  # Tekst titla
                
                # Pripremi podatke za prijevod
                subtitle = {
                    'text': original_text,
                    'translation_info': {}
                }
                
                # Obradi disklejmere
                context = self.translator._get_surrounding_context(i, [subtitle])
                processed_text = self.translator._process_discourse_marker(
                    text=original_text, 
                    context=context
                )
                
                # Provjeri je li rezultat očekivan
                if i < len(expected_results):
                    print(f"\nTest {i+1}:")
                    print(f"  Original: {original_text}")
                    print(f"  Očekivano: {expected_results[i]}")
                    print(f"  Dobiveno: {processed_text}")
                    
                    # Provjeri je li rezultat barem 50% sličan očekivanom
                    similarity = self._calculate_similarity(processed_text, expected_results[i])
                    self.assertGreaterEqual(
                        similarity, 0.5,
                        f"Prevod '{processed_text}' nije dovoljno sličan očekivanom '{expected_results[i]}'"
                    )
                    print(f"  Sličnost: {similarity*100:.1f}%")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Izračunava sličnost između dva teksta (0-1)."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
