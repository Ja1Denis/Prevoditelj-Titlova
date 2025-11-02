import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from transformers import MarianMTModel, MarianTokenizer

@dataclass
class DialogueContext:
    """Klasa za praćenje konteksta dijaloga i roda govornika."""
    speaker_gender: Dict[str, str] = field(default_factory=dict)
    current_speaker: Optional[str] = None
    previous_lines: List[str] = field(default_factory=list)
    next_lines: List[str] = field(default_factory=list)
    context_window: int = 3  # Koliko rečenica prije/poslije gledamo za kontekst

    def update_context(self, current_idx: int, all_subtitles: List[dict]):
        """Ažurira kontekst s okolnim titlovima."""
        start_idx = max(0, current_idx - self.context_window)
        end_idx = min(len(all_subtitles), current_idx + self.context_window + 1)
        
        self.previous_lines = [sub['text'] for sub in all_subtitles[start_idx:current_idx]]
        self.next_lines = [sub['text'] for sub in all_subtitles[current_idx + 1:end_idx]]

    def add_speaker_gender(self, speaker: str, gender: str):
        """Dodaje ili ažurira rod govornika."""
        self.speaker_gender[speaker.lower()] = gender

    def get_speaker_gender(self, speaker: str) -> Optional[str]:
        """Dohvaća rod govornika ako je poznat."""
        return self.speaker_gender.get(speaker.lower())

class SubtitleAnalyzer:
    """Klasa za analizu titlova i detekciju konteksta."""
    
    def __init__(self):
        # Rodno neutralne imenice i njihovi prijevodi
        self.neutral_nouns = {
            'friend': ('prijatelj', 'prijateljica'),
            'teacher': ('učitelj', 'učiteljica'),
            'doctor': ('liječnik', 'liječnica'),
            'neighbor': ('susjed', 'susjeda'),
            'student': ('student', 'studentica'),
            'professor': ('profesor', 'profesorica'),
            'writer': ('pisac', 'spisateljica'),
            'driver': ('vozač', 'vozačica'),
            'worker': ('radnik', 'radnica'),
            'artist': ('umjetnik', 'umjetnica'),
            'actor': ('glumac', 'glumica'),
            'singer': ('pjevač', 'pjevačica'),
            'dancer': ('plesač', 'plesačica'),
            'assistant': ('asistent', 'asistentica'),
            'manager': ('upravitelj', 'upraviteljica'),
            'colleague': ('kolega', 'kolegica')
        }
        
        # Indikatori roda
        self.female_indicators = {
            'zamjenice': ['she', 'her', 'hers', 'herself'],
            'titule': ['lady', 'queen', 'princess', 'mrs', 'ms', 'miss', 'madam'],
            'odnosi': ['mother', 'daughter', 'sister', 'aunt', 'grandmother', 'wife', 'girlfriend'],
            'pridjevi': ['beautiful', 'pregnant', 'female'],
            'imena': ['mary', 'elizabeth', 'sarah', 'emma', 'sophia']
        }
        
        self.male_indicators = {
            'zamjenice': ['he', 'his', 'him', 'himself'],
            'titule': ['lord', 'king', 'prince', 'mr', 'sir'],
            'odnosi': ['father', 'son', 'brother', 'uncle', 'grandfather', 'husband', 'boyfriend'],
            'pridjevi': ['handsome', 'male'],
            'imena': ['john', 'william', 'james', 'robert', 'michael']
        }
        
        # Pravila za množinu
        self.plural_rules = {
            'female_only': {
                'indicators': [
                    r'(?:girl|woman|lady|mother|daughter|sister)s?\s+and\s+(?:girl|woman|lady|mother|daughter|sister)s?.*?they',
                    r'both\s+(?:girl|woman|lady|mother|daughter|sister)s.*?they',
                    r'the\s+(?:girl|woman|lady|mother|daughter|sister)s.*?they'
                ],
                'translation': 'one'  # za "one su"
            },
            'neutral': {
                'indicators': [
                    r'(?:child|kid)(?:ren)?.*?they',
                    r'the\s+(?:child|kid)(?:ren)?.*?they'
                ],
                'translation': 'ona'  # za "ona su"
            }
            # Ako nije ni jedno od gore navedenog, koristi "oni"
        }
        
        # Glagoli koji često otkrivaju rod
        self.gendered_verbs = [
            'was', 'were', 'had', 'did', 'went', 'came', 'said', 'told',
            'felt', 'thought', 'knew', 'wanted', 'needed', 'liked'
        ]

    def analyze_text(self, text: str, context: DialogueContext) -> Tuple[str, List[str]]:
        """
        Analizira tekst i vraća informacije o rodu i dodatne kontekstualne oznake.
        """
        text_lower = text.lower()
        detected_gender = None
        context_markers = []
        
        # 1. Provjeri za eksplicitnog govornika
        speaker_match = re.match(r'^([^:]+):', text)
        if speaker_match:
            speaker = speaker_match.group(1).strip()
            known_gender = context.get_speaker_gender(speaker)
            if known_gender:
                detected_gender = known_gender
            context.current_speaker = speaker
        
        # 2. Traži indikatore roda u trenutnom tekstu
        if not detected_gender:
            # Provjeri ženske indikatore
            for category in self.female_indicators:
                if any(indicator in text_lower for indicator in self.female_indicators[category]):
                    detected_gender = "female"
                    break
            
            # Provjeri muške indikatore
            if not detected_gender:
                for category in self.male_indicators:
                    if any(indicator in text_lower for indicator in self.male_indicators[category]):
                        detected_gender = "male"
                        break
        
        # 3. Ako još nemamo rod, provjeri kontekst
        if not detected_gender:
            # Provjeri prethodne titlove
            for prev_text in context.previous_lines:
                gender = self._check_gender_indicators(prev_text.lower())
                if gender:
                    detected_gender = gender
                    break
        
        # 4. Dodaj oznake za posebne slučajeve
        # Glagolsko vrijeme
        if any(verb in text_lower.split() for verb in self.gendered_verbs):
            context_markers.append("PAST")
        
        # Pridjevi
        if re.search(r'\b(happy|sad|angry|tired|beautiful|handsome|smart|good|bad)\b', text_lower):
            context_markers.append("ADJ")
        
        # Direktni govor
        if '"' in text or "'" in text:
            context_markers.append("QUOTE")
        
        # 5. Zapamti rod govornika ako je otkriven
        if detected_gender and context.current_speaker:
            context.add_speaker_gender(context.current_speaker, detected_gender)
        
        return f"<{detected_gender}>" if detected_gender else "", context_markers

    def _check_gender_indicators(self, text: str) -> Optional[str]:
        """Pomoćna metoda za provjeru indikatora roda u tekstu."""
        text_lower = text.lower()
        
        # Prvo provjeri za rodno neutralne imenice
        for neutral_word, (male_form, female_form) in self.neutral_nouns.items():
            if neutral_word in text_lower:
                # Provjeri kontekst za rod
                if any(indicator in text_lower for cat in self.female_indicators.values() for indicator in cat):
                    return "female"
                elif any(indicator in text_lower for cat in self.male_indicators.values() for indicator in cat):
                    return "male"
        
        # Zatim provjeri ostale indikatore
        for category in self.female_indicators:
            if any(indicator in text_lower for indicator in self.female_indicators[category]):
                return "female"
        for category in self.male_indicators:
            if any(indicator in text_lower for indicator in self.male_indicators[category]):
                return "male"
        return None
        
    def _analyze_plural(self, text: str, context_lines: List[str]) -> Optional[str]:
        """Analizira množinu i određuje prijevod za 'they'."""
        text_lower = ' '.join([text.lower()] + [line.lower() for line in context_lines])
        
        # Provjeri pravila za žensku množinu
        for pattern in self.plural_rules['female_only']['indicators']:
            if re.search(pattern, text_lower):
                return "female_plural"
                
        # Provjeri pravila za srednji rod
        for pattern in self.plural_rules['neutral']['indicators']:
            if re.search(pattern, text_lower):
                return "neutral_plural"
        
        # Ako nema specifičnog pravila, vrati None (koristit će se "oni")
        return None
        
    def _get_gender_specific_translation(self, text: str, detected_gender: str) -> str:
        """Prilagođava prijevod na temelju roda."""
        text_lower = text.lower()
        
        # Zamijeni rodno neutralne imenice s odgovarajućim oblikom
        for eng_word, (male_form, female_form) in self.neutral_nouns.items():
            if eng_word in text_lower:
                return text.replace(eng_word, female_form if detected_gender == "female" else male_form)
        
        return text

class SubtitleTranslator:
    """Glavna klasa za prevođenje titlova."""
    
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-zls"):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.analyzer = SubtitleAnalyzer()
        self.context = DialogueContext()
        
    def translate_file(self, input_path: str, batch_size: int = 3):
        """Prevodi cijelu datoteku s titlovima."""
        input_path = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}.hr{input_path.suffix}"
        
        if output_path.exists():
            print(f"⚠️  Preskačem - već postoji: {output_path.name}")
            return False
        
        # Učitaj titlove
        subtitles = self._parse_srt(input_path)
        if not subtitles:
            print("❌ Nema titlova za prevođenje")
            return False
        
        print(f"📄 Prevođenje: {input_path.name}")
        print(f"   Pronađeno {len(subtitles)} titlova")
        
        # Prevedi u grupama
        i = 0
        while i < len(subtitles):
            batch_end = min(i + batch_size, len(subtitles))
            batch_subtitles = subtitles[i:batch_end]
            
            # Ažuriraj kontekst za svaki titl u batchu
            for j, sub in enumerate(batch_subtitles):
                self.context.update_context(i + j, subtitles)
                gender_token, markers = self.analyzer.analyze_text(sub['text'], self.context)
                
                # Pripremi tekst za prevođenje s kontekstom
                context_info = f"[{' '.join(markers)}] " if markers else ""
                sub['translation_text'] = f">>hrv<< {gender_token} {context_info}{sub['text']}"
            
            # Prevedi batch
            batch_texts = [sub['translation_text'] for sub in batch_subtitles]
            translations = self._translate_batch(batch_texts)
            
            # Spremi prijevode
            for j, translation in enumerate(translations):
                subtitles[i + j]['text'] = translation
            
            i = batch_end
            if i % 25 == 0:
                print(f"   Prevedeno {i}/{len(subtitles)}...")
        
        # Spremi prevedene titlove
        self._write_srt(subtitles, output_path)
        print(f"✓ Spremljeno kao: {output_path.name}")
        return True

    def _parse_srt(self, file_path: str) -> List[dict]:
        """Parsira SRT datoteku."""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1250', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"❌ Greška: Ne mogu pročitati file {file_path}. Pokušajte ručno pretvoriti u UTF-8 format.")
            return []
        
        blocks = re.split(r'\n\n+', content.strip())
        subtitles = []
        
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                subtitles.append({
                    'index': lines[0].strip(),
                    'timestamp': lines[1].strip(),
                    'text': '\n'.join(lines[2:])
                })
        
        return subtitles

    def _translate_batch(self, texts: List[str]) -> List[str]:
        """Prevodi grupu tekstova."""
        try:
            inputs = self.tokenizer(texts, 
                                  return_tensors="pt", 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512)
            
            translations = self.model.generate(**inputs, max_length=512)
            translated_texts = [self.tokenizer.decode(t, skip_special_tokens=True) 
                              for t in translations]
            
            # Očisti prijevode
            cleaned_texts = []
            for text in translated_texts:
                # Ukloni oznake
                text = re.sub(r'\s*<(male|female)>\s*', ' ', text)
                text = re.sub(r'\s*\[[^\]]*\]\s*', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                cleaned_texts.append(text.strip())
            
            return cleaned_texts
            
        except Exception as e:
            print(f"⚠️  Greška u prijevodu: {e}")
            return texts  # Vrati originalne tekstove u slučaju greške

    def _write_srt(self, subtitles: List[dict], output_path: str):
        """Zapisuje titlove u SRT format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subtitles):
                f.write(f"{sub['index']}\n")
                f.write(f"{sub['timestamp']}\n")
                f.write(f"{sub['text']}\n")
                if i < len(subtitles) - 1:
                    f.write("\n")

def main():
    # Unos putanje
    folder_path = input("Unesi putanju do foldera s SRT titlovima: ").strip().strip('"')
    
    if not os.path.exists(folder_path):
        print(f"❌ Greška: Folder '{folder_path}' ne postoji!")
        return
    
    # Pronađi SRT datoteke
    folder = Path(folder_path)
    all_srt_files = list(folder.glob("*.srt"))
    print("\nPronađene SRT datoteke:")
    for f in all_srt_files:
        print(f"  - {f.name}")
    
    # Filtriraj već prevedene
    srt_files = [f for f in all_srt_files 
                 if not f.stem.endswith('.hr') 
                 and not f.stem.endswith('.HR')
                 and 'hrv' not in f.stem.lower() 
                 and 'cro' not in f.stem.lower()]
    
    print("\nDatoteke za prevođenje (nakon filtriranja):")
    for f in srt_files:
        print(f"  - {f.name}")
    
    if not srt_files:
        print("❌ Nema novih fileova za prevođenje")
        return
    
    print(f"\n📁 Pronađeno {len(srt_files)} SRT file(ova) za prevođenje\n")
    
    # Inicijaliziraj prevoditelj
    print("🔄 Učitavanje modela (ovo može potrajati pri prvom pokretanju)...")
    translator = SubtitleTranslator()
    print("✓ Model učitan!\n")
    
    # Prevedi sve datoteke
    translated_count = 0
    for i, srt_file in enumerate(srt_files, 1):
        print(f"[{i}/{len(srt_files)}]")
        if translator.translate_file(str(srt_file)):
            translated_count += 1
    
    print(f"\n{'='*50}")
    print(f"✓ GOTOVO! Prevedeno {translated_count}/{len(srt_files)} fileova")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()