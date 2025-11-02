import re
import os
import csv
import time
import requests
import difflib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from transformers import MarianMTModel, MarianTokenizer
import threading

@dataclass
class DialogueContext:
    """Klasa za praćenje konteksta dijaloga."""
    speaker_gender: Dict[str, str] = field(default_factory=dict)
    speaker_history: Dict[str, List[str]] = field(default_factory=dict)
    current_speaker: Optional[str] = None
    previous_lines: List[str] = field(default_factory=list)
    next_lines: List[str] = field(default_factory=list)
    context_window: int = 3
    known_female_speakers: Set[str] = field(default_factory=set)
    known_male_speakers: Set[str] = field(default_factory=set)

    def update_context(self, current_idx: int, all_subtitles: List[dict]):
        """Ažurira kontekst s okolnim titlovima."""
        start_idx = max(0, current_idx - self.context_window)
        end_idx = min(len(all_subtitles), current_idx + self.context_window + 1)
        
        print(f"\nAžuriram kontekst za titl {current_idx+1}...")
        self.previous_lines = [sub['text'] for sub in all_subtitles[start_idx:current_idx]]
        self.next_lines = [sub['text'] for sub in all_subtitles[current_idx + 1:end_idx]]

    def add_speaker_line(self, speaker: str, line: str):
        """Dodaje novu repliku u povijest govornika."""
        if speaker not in self.speaker_history:
            self.speaker_history[speaker] = []
        self.speaker_history[speaker].append(line)

    def get_speaker_history(self, speaker: str) -> List[str]:
        """Dohvaća povijest replika govornika."""
        return self.speaker_history.get(speaker, [])

    def set_speaker_gender(self, speaker: str, gender: str):
        """Postavlja rod govornika."""
        speaker = speaker.lower()
        self.speaker_gender[speaker] = gender
        if gender == 'female':
            self.known_female_speakers.add(speaker)
        elif gender == 'male':
            self.known_male_speakers.add(speaker)

    def get_speaker_gender(self, speaker: str) -> Optional[str]:
        """Dohvaća rod govornika ako je poznat."""
        return self.speaker_gender.get(speaker.lower())

class ConversationAnalyzer:
    """Klasa za analizu konverzacija između govornika."""
    
    def __init__(self):
        self.speaker_interactions: Dict[str, List[str]] = {}
        self.scene_contexts: List[Dict] = []
        self.current_scene_speakers: Set[str] = set()
        
    def add_speaker_line(self, speaker: str, gender: Optional[str], text: str):
        """Dodaje liniju govornika."""
        if speaker not in self.speaker_interactions:
            self.speaker_interactions[speaker] = []
        self.speaker_interactions[speaker].append({
            'text': text,
            'gender': gender
        })
        self.current_scene_speakers.add(speaker)
    
    def update_scene_context(self, speaker: str, gender: str):
        """Ažurira kontekst scene."""
        self.scene_contexts.append({
            'speaker': speaker,
            'gender': gender,
            'speakers_in_scene': list(self.current_scene_speakers)
        })
    
    def get_context_for_speaker(self, speaker: str) -> Dict:
        """Dohvaća kontekst za govornika."""
        return {
            'conversation_type': 'unknown',
            'is_response_to': False,
            'previous_speakers_gender': None
        }
    
    def suggest_gender(self, speaker: str) -> Optional[str]:
        """Sugerira rod govornika na temelju povijesti."""
        if speaker not in self.speaker_interactions:
            return None
        
        lines = self.speaker_interactions[speaker]
        genders = [line['gender'] for line in lines if line['gender']]
        
        if not genders:
            return None
        
        # Vrati najčešći rod
        female_count = genders.count('female')
        male_count = genders.count('male')
        
        if female_count > male_count:
            return 'female'
        elif male_count > female_count:
            return 'male'
        
        return None

class GenderAnalyzer:
    """Klasa za analizu i određivanje roda."""
    
    def __init__(self):
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
        
        self.female_indicators = {
            'zamjenice': ['she', 'her', 'hers', 'herself'],
            'titule': ['lady', 'queen', 'princess', 'mrs', 'ms', 'miss', 'madam'],
            'odnosi': ['mother', 'daughter', 'sister', 'aunt', 'grandmother', 'wife', 'girlfriend'],
            'pridjevi': ['beautiful', 'pregnant', 'female'],
            'glagoli': ['dressed', 'pregnant']
        }
        
        self.male_indicators = {
            'zamjenice': ['he', 'his', 'him', 'himself'],
            'titule': ['lord', 'king', 'prince', 'mr', 'sir'],
            'odnosi': ['father', 'son', 'brother', 'uncle', 'grandfather', 'husband', 'boyfriend'],
            'pridjevi': ['handsome', 'male'],
            'glagoli': ['shaved', 'bearded']
        }
        
        self.plural_patterns = {
            'female_only': [
                (r'(?:girl|woman|lady|mother|daughter|sister)s?\s+and\s+(?:girl|woman|lady|mother|daughter|sister)s?.*?they', 'one'),
                (r'both\s+(?:girl|woman|lady|mother|daughter|sister)s.*?they', 'one'),
                (r'the\s+(?:girl|woman|lady|mother|daughter|sister)s.*?they', 'one')
            ],
            'neutral': [
                (r'(?:child|kid)(?:ren)?.*?they', 'ona'),
                (r'the\s+(?:child|kid)(?:ren)?.*?they', 'ona')
            ]
        }

class ImprovedSubtitleTranslator:
    """Glavna klasa za prevođenje titlova."""
    
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-zls", metadata_csv: Optional[str] = None, user_dict_path: Optional[str] = None):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.gender_analyzer = GenderAnalyzer()
        self.context = DialogueContext()
        self.conversation_analyzer = ConversationAnalyzer()
        self._speaker_gender_external: Dict[str, str] = {}
        self._csv_text_gender: Dict[str, str] = {}  # normalizirani eng. tekst -> gender govornika
        self._name_gender_map: Dict[str, str] = {}
        self._user_pairs: List[Tuple[re.Pattern, str, int]] = []  # (pattern, replacement, priority)
        
        # Učitaj discourse markers UVIJEK (prije user dict)
        self._load_discourse_markers()
        
        # Učitaj CSV metapodatke
        if metadata_csv and os.path.exists(metadata_csv):
            try:
                with open(metadata_csv, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        speaker = (row.get('speaker') or row.get('govornik') or '').strip().lower()
                        gender = (row.get('gender') or row.get('spol') or '').strip().lower()
                        text_en = (row.get('text') or row.get('line') or row.get('utterance') or '').strip()
                        if speaker and gender:
                            self._speaker_gender_external[speaker] = 'female' if gender.startswith(('f','ž')) else 'male'
                        if text_en:
                            norm = self._normalize_eng(text_en)
                            if norm and gender:
                                self._csv_text_gender[norm] = 'female' if gender.startswith(('f','ž')) else 'male'
            except Exception:
                self._speaker_gender_external = {}
                self._csv_text_gender = {}
        
        # Učitaj korisnički rječnik (opcionalno) - NAKON discourse markers
        if user_dict_path and os.path.exists(user_dict_path):
            try:
                self._load_user_dictionary(user_dict_path)
            except Exception:
                pass
        
        # Učitaj CSV s imenima i rodom (npr. "imena_kategorizirana_ispravljeno.csv")
        if metadata_csv and os.path.exists(metadata_csv):
            try:
                self._name_gender_map = self._load_name_gender_csv(metadata_csv)
            except Exception:
                self._name_gender_map = {}
        # Učitaj .env varijable (python-dotenv ako postoji; fallback ručno)
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except Exception:
            env_path = os.path.join(os.getcwd(), '.env')
            try:
                if os.path.exists(env_path):
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            if '=' in line:
                                k, v = line.split('=', 1)
                                k = k.strip()
                                v = v.strip().strip('"').strip("'")
                                if k and k not in os.environ:
                                    os.environ[k] = v
            except Exception:
                pass
        # Gemini postavke iz okoline (bez GUI-a)
        self._gem_api_key = os.getenv('GOOGLE_API_KEY', '').strip()
        self._gem_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash').strip()
        try:
            self._gem_temp = float(os.getenv('GEMINI_TEMPERATURE', '0.3'))
        except Exception:
            self._gem_temp = 0.3
        try:
            self._gem_max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', '120'))
        except Exception:
            self._gem_max_tokens = 120
        try:
            self._gem_timeout = int(os.getenv('GEMINI_TIMEOUT', '12'))
        except Exception:
            self._gem_timeout = 12
        # Enable/disable preko env: GEMINI_ENABLE=1 (default 0 -> isključeno)
        self._gem_enabled = os.getenv('GEMINI_ENABLE', '0').strip() == '1'

    def _load_name_gender_csv(self, path: str) -> Dict[str, str]:
        """Učitava CSV s imenima i rodom."""
        name_gender_map = {}
        with open(path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get('name') or row.get('ime') or '').strip().lower()
                gender = (row.get('gender') or row.get('spol') or '').strip().lower()
                if name and gender:
                    name_gender_map[name] = 'female' if gender.startswith(('f','ž')) else 'male'
        return name_gender_map

    def _parse_timestamp(self, timestamp: str) -> datetime:
        """Parsira SRT timestamp u datetime objekt."""
        # Format: 00:00:20,000 ili 00:00:20.000
        timestamp = timestamp.strip().replace(',', '.')
        
        try:
            # Parsiranje formata HH:MM:SS.mmm
            parts = timestamp.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split('.')
            seconds = int(seconds_parts[0])
            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
            
            # Kreiraj datetime objekt (koristimo bilo koji datum, važno je samo vrijeme)
            base_date = datetime(2000, 1, 1)
            return base_date + timedelta(
                hours=hours,
                minutes=minutes,
                seconds=seconds,
                milliseconds=milliseconds
            )
        except Exception as e:
            print(f"⚠️  Greška pri parsiranju timestampa '{timestamp}': {e}")
            return datetime(2000, 1, 1)

    def translate_file(self, input_path: str, output_path: str = None) -> bool:
        """Glavna metoda za prevođenje SRT datoteke s automatskim preimenovanjem."""
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"❌ Datoteka ne postoji: {input_path}")
            return False
        
        # Definiraj dodatne direktorije za pretraživanje
        search_dirs = [
            input_path.parent,  # Direktorij s izvornim SRT-om
            input_path.parent / "Prevedeno",  # Poddirektorij za prevedene titlove
            input_path.parent.parent,  # Roditeljski direktorij
        ]
        
        # Pronađi odgovarajuću video datoteku
        print(f"🔍 Tražim video datoteku u: {', '.join(str(d) for d in search_dirs if d.exists())}")
        video_file = self._find_matching_video_file(str(input_path), search_dirs)
        
        if not video_file:
            print(f"⚠️  Nije pronađena odgovarajuća video datoteka za {input_path.name}")
            if not output_path:
                output_path = input_path.with_suffix('.hr.srt')
        else:
            print(f"✅ Pronađena odgovarajuća video datoteka: {video_file.name}")
            if not output_path:
                output_path = self._generate_output_filename(video_file, input_path)
        
        output_path = Path(output_path)
        
        # Ako izlazna datoteka već postoji, preskoči
        if output_path.exists():
            print(f"⚠️  Preskačem - već postoji: {output_path.name}")
            return False
        
        print(f"📄 Prevođenje: {input_path.name} -> {output_path.name}")
        
        subtitles = self._parse_srt(input_path)
        if not subtitles:
            print("❌ Nema titlova za prevođenje")
            return False
        
        print(f"   Pronađeno {len(subtitles)} titlova")
        
        # 1. Prva analiza konteksta dijaloga
        self._analyze_dialogue_context(subtitles)
        
        # 2. Prva obrada titlova s analizom konteksta
        for i, subtitle in enumerate(subtitles):
            self.context.update_context(i, subtitles)
            subtitle['translation_info'] = self._analyze_subtitle(subtitle['text'], i, subtitles)
        
        # 3. Primijeni diskursne markere i druge prijevode (PRE-PROCESSING)
        # OVO JE KLJUČNI KORAK ZA ISPRAVAK "Well..." -> "Pa..."
        print("\nObrada disklejmera...")
        for i, subtitle in enumerate(subtitles):
            text = subtitle['text']
            # Primijeni diskursne markere na originalni tekst
            if text:
                # Spremi originalni tekst prije primjene diskursnih markera
                original_text = text
                
                # Primijeni diskursne markere i lažne prijatelje
                # Ovdje se "Well, I don't know" pretvara u "Pa, I don't know"
                text = self._apply_false_friends(text)
                
                # Ako je došlo do promjena, ažuriraj tekst i ponovno analiziraj
                if text != original_text:
                    subtitle['text'] = text
                    # Ažuriraj kontekst s novim tekstom
                    self.context.update_context(i, subtitles)
                    subtitle['translation_info'] = self._analyze_subtitle(text, i, subtitles)
        
        # 4. Sada možemo početi s prijevodom
        translated_count = 0
        batch_size = 3
        
        for i in range(0, len(subtitles), batch_size):
            batch_end = min(i + batch_size, len(subtitles))
            batch = subtitles[i:batch_end]
            
            for j, subtitle in enumerate(batch):
                global_idx = i + j
                self.context.update_context(global_idx, subtitles)
                # Ako već nismo analizirali ovaj titl u prethodnom koraku
                if 'translation_info' not in subtitle:
                    # Analiziramo tekst koji je možda već polu-preveden (npr. "Pa, I don't know")
                    subtitle['translation_info'] = self._analyze_subtitle(subtitle['text'], global_idx, subtitles)
            
            print(f"\nPokrećem batch prevođenje {i+1}-{batch_end}...")
            self._translate_batch(batch)
            translated_count += len(batch)
            
            if translated_count % 25 == 0:
                print(f"   Prevedeno {translated_count}/{len(subtitles)}...")
        
        self._write_srt(subtitles, output_path)
        print(f"✓ Spremljeno kao: {output_path.name}")
        return True

    def _analyze_dialogue_context(self, subtitles: List[dict]):
        """Prvi prolaz kroz titlove za analizu konteksta dijaloga."""
        print("   Analiziram kontekst dijaloga...")
        
        current_scene = []
        scenes = []
        
        for i, subtitle in enumerate(subtitles):
            text = subtitle['text']
            
            if i > 0:
                prev_end = self._parse_timestamp(subtitles[i-1]['timestamp'].split(' --> ')[1])
                curr_start = self._parse_timestamp(subtitle['timestamp'].split(' --> ')[0])
                if (curr_start - prev_end).total_seconds() > 5:
                    if current_scene:
                        scenes.append(current_scene)
                        current_scene = []
            
            current_scene.append(subtitle)
        
        if current_scene:
            scenes.append(current_scene)
        
        for scene in scenes:
            scene_speakers = set()
            scene_gender_hints = {'female': 0, 'male': 0}
            
            for subtitle in scene:
                text = subtitle['text'].lower()
                
                if 'girl' in text or 'woman' in text or 'lady' in text:
                    scene_gender_hints['female'] += 1
                if 'boy' in text or 'man' in text or 'gentleman' in text:
                    scene_gender_hints['male'] += 1
                
                speaker_match = re.match(r'^([^:]+):', subtitle['text'])
                if speaker_match:
                    speaker = speaker_match.group(1).strip().lower()
                    scene_speakers.add(speaker)
                    
                    numbered_speaker_match = re.match(r'^(girl|boy|woman|man)\s*\d+', speaker, re.IGNORECASE)
                    if numbered_speaker_match:
                        base_speaker = numbered_speaker_match.group(1).lower()
                        if base_speaker in ['girl', 'woman']:
                            scene_gender_hints['female'] += 2
                        elif base_speaker in ['boy', 'man']:
                            scene_gender_hints['male'] += 2
            
            scene_type = None
            if scene_gender_hints['female'] > scene_gender_hints['male'] * 2:
                scene_type = 'female_only'
            elif scene_gender_hints['male'] > scene_gender_hints['female'] * 2:
                scene_type = 'male_only'
            
            for subtitle in scene:
                text = subtitle['text']
                speaker_match = re.match(r'^([^:]+):', text)
                
                if speaker_match:
                    speaker = speaker_match.group(1).strip().lower()
                    self.context.add_speaker_line(speaker, text)
                    
                    known_gender = self.context.get_speaker_gender(speaker)
                    if known_gender:
                        if scene_type and known_gender != scene_type.replace('_only', ''):
                            self.conversation_analyzer.update_scene_context(speaker, known_gender)
                        continue
                    
                    if scene_type:
                        self.context.set_speaker_gender(
                            speaker, 
                            'female' if scene_type == 'female_only' else 'male'
                        )
                        continue
                    
                    speaker_lines = self.context.get_speaker_history(speaker)
                    combined_text = ' '.join(speaker_lines)
                    
                    female_indicators = sum(
                        1 for category in self.gender_analyzer.female_indicators
                        for indicator in self.gender_analyzer.female_indicators[category]
                        if indicator in combined_text.lower()
                    )
                    
                    male_indicators = sum(
                        1 for category in self.gender_analyzer.male_indicators
                        for indicator in self.gender_analyzer.male_indicators[category]
                        if indicator in combined_text.lower()
                    )
                    
                    if female_indicators > male_indicators:
                        self.context.set_speaker_gender(speaker, 'female')
                    elif male_indicators > female_indicators:
                        self.context.set_speaker_gender(speaker, 'male')

    def _analyze_subtitle(self, text: str, current_idx: int, all_subtitles: List[dict]) -> Dict:
        """Analizira pojedinačni titl za prevođenje."""
        info = {
            'gender': None,
            'plural_type': None,
            'neutral_nouns': [],
            'context_markers': [],
            'conversation_type': None
        }
        
        text_lower = text.lower()
        current_speaker = None
        
        speaker_match = re.match(r'^([^:]+):', text)
        if speaker_match:
            current_speaker = speaker_match.group(1).strip().lower()
        
        if current_speaker:
            ext_gender = self._speaker_gender_external.get(current_speaker)
            if ext_gender:
                info['gender'] = ext_gender
            conversation_context = self.conversation_analyzer.get_context_for_speaker(current_speaker)
            info['conversation_type'] = conversation_context['conversation_type']
            
            if conversation_context['is_response_to']:
                info['context_markers'].append('RESPONSE')
                if conversation_context['previous_speakers_gender']:
                    info['context_markers'].append(f"RESPONDING_TO_{conversation_context['previous_speakers_gender'].upper()}")
            
            if conversation_context['conversation_type'] != 'unknown':
                info['context_markers'].append(f"CONV_{conversation_context['conversation_type'].upper()}")
        
        # Analiza 'they' se mora dogoditi na originalnom engleskom tekstu,
        # ali 'text' ovdje može biti npr. "Pa, I don't know".
        # Stoga, ova analiza možda neće biti pouzdana ako se 'they' pojavi
        # u rečenici koja počinje s discourse markerom.
        # Za sada ostavljamo, ali ovo je potencijalno slabo mjesto.
        they_pattern = r'\b(they|them|their)\b'
        if re.search(they_pattern, text_lower):
            if info['conversation_type'] == 'female_only':
                info['plural_type'] = 'one'
            elif info['conversation_type'] == 'male_only':
                info['plural_type'] = 'oni'
            else:
                for pattern_type, patterns in self.gender_analyzer.plural_patterns.items():
                    for pattern, plural_type in patterns:
                        if re.search(pattern, text_lower):
                            info['plural_type'] = plural_type
                            break
                    if info['plural_type']:
                        break
                
                if not info['plural_type']:
                    info['plural_type'] = 'oni'
        
        for noun in self.gender_analyzer.neutral_nouns:
            if noun in text_lower:
                info['neutral_nouns'].append(noun)
        
        # Ako je poznat govornik, prvo pokušaj iz vanjskog CSV-a imena->rod
        if current_speaker:
            # Pokušaj direktan lookup po imenu (case-insensitive)
            g = self._name_gender_map.get(current_speaker.lower())
            if g in ('female', 'male'):
                info['gender'] = g
                info['context_markers'].append('NAME_CSV')
            # Ako nije u CSV-u, pokušaj iz memorije konteksta
            if not info['gender']:
                info['gender'] = self.context.get_speaker_gender(current_speaker)
            
            if not info['gender']:
                suggested_gender = self.conversation_analyzer.suggest_gender(current_speaker)
                if suggested_gender:
                    info['gender'] = suggested_gender
                    info['context_markers'].append('SUGGESTED_GENDER')
        
        if not info['gender']:
            for category in self.gender_analyzer.female_indicators:
                if any(indicator in text_lower 
                      for indicator in self.gender_analyzer.female_indicators[category]):
                    info['gender'] = 'female'
                    break
            
            if not info['gender']:
                for category in self.gender_analyzer.male_indicators:
                    if any(indicator in text_lower 
                          for indicator in self.gender_analyzer.male_indicators[category]):
                        info['gender'] = 'male'
        
        # Vrati info. info['src_text'] će biti postavljen u _translate_batch
        return info

    def _prepare_text_for_translation(self, text: str, info: dict) -> str:
        """Priprema tekst za prijevod dodavanjem oznaka jezika, roda i množine."""
        translation_text = ">>hrv<< "
        # Ako postoji rod govornika, dodaj oznaku
        if info.get('gender'):
            translation_text += f"<{info['gender']}> "
        # Ako postoji množinski oblik, dodaj oznaku
        if info.get('plural_type'):
            translation_text += f"[PLURAL_{info['plural_type'].upper()}] "
        # Ako postoje neutralne imenice, zamijeni ih prema rodu
        if info.get('gender') and info.get('neutral_nouns'):
            for noun in info['neutral_nouns']:
                if noun in self.gender_analyzer.neutral_nouns:
                    male_form, female_form = self.gender_analyzer.neutral_nouns[noun]
                    replacement = female_form if info['gender'] == 'female' else male_form
                    text = re.sub(rf"\b{re.escape(noun)}\b", replacement, text, flags=re.IGNORECASE)
        
        # 'text' ovdje može biti npr. "Pa, I don't know"
        return translation_text + text

    def _load_name_gender_csv(self, path: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        try:
            with open(path, 'r', encoding='utf-8', newline='') as f:
                sample = f.read(4096)
                f.seek(0)
                delimiter = ','
                if '\t' in sample:
                    delimiter = '\t'
                else:
                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
                        delimiter = dialect.delimiter
                    except Exception:
                        delimiter = ','
                reader = csv.reader(f, delimiter=delimiter)
                header = next(reader, None)
                name_idx = None
                gender_idx = None
                if header:
                    lower = [str(h).strip().lower() for h in header]
                    for i, h in enumerate(lower):
                        if h in ('name', 'ime', 'speaker', 'govornik', 'ame') and name_idx is None:
                            name_idx = i
                        if h in ('gender', 'rod', 'spol') and gender_idx is None:
                            gender_idx = i
                if name_idx is None or gender_idx is None:
                    # Bez pouzdanog zaglavlja: pretpostavi da su prve dvije kolone ime i rod
                    if header:
                        # Obradi i header kao običan red ako nema jasnih indeksa
                        row = header
                        if len(row) >= 2:
                            name = str(row[0]).strip()
                            gen = str(row[1]).strip().lower().strip(' .,!?:;')
                            if name:
                                if gen in ('female', 'žensko', 'zensko', 'f'):
                                    mapping[name.lower()] = 'female'
                                elif gen in ('male', 'muško', 'musko', 'm'):
                                    mapping[name.lower()] = 'male'
                    for row in reader:
                        if not row or len(row) < 2:
                            continue
                        name = str(row[0]).strip()
                        gen = str(row[1]).strip().lower().strip(' .,!?:;')
                        if not name:
                            continue
                        if gen in ('nije_ime', 'nepoznato', 'unknown', ''):
                            continue
                        if gen in ('female', 'žensko', 'zensko', 'f'):
                            mapping[name.lower()] = 'female'
                        elif gen in ('male', 'muško', 'musko', 'm'):
                            mapping[name.lower()] = 'male'
                    return mapping
                # Inače koristi pronađene indekse
                for row in reader:
                    if not row:
                        continue
                    if max(name_idx, gender_idx) >= len(row):
                        continue
                    name = str(row[name_idx]).strip()
                    gen = str(row[gender_idx]).strip().lower().strip(' .,!?:;')
                    if not name:
                        continue
                    if gen in ('nije_ime', 'nepoznato', 'unknown', ''):
                        continue
                    if gen in ('female', 'žensko', 'zensko', 'f'):
                        mapping[name.lower()] = 'female'
                    elif gen in ('male', 'muško', 'musko', 'm'):
                        mapping[name.lower()] = 'male'
        except Exception:
            return {}
        return mapping

    def _translate_batch(self, subtitles: List[dict]):
        """Prevodi grupu titlova s kontekstom."""
        print("\n=== ANALIZA KONTEKSTA PRIJE PREVOĐENJA ===")
        for i, sub in enumerate(subtitles):
            print(f"\nTitl {i+1}:")
            print(f"  Tekst: {sub['text']}")
            if 'translation_info' in sub and 'context' in sub['translation_info']:
                ctx = sub['translation_info']['context']
                print(f"  Prethodni kontekst: {ctx.get('prev_texts', [])}")
                print(f"  Sljedeći kontekst: {ctx.get('next_texts', [])}")
        print("=== KRAJ ANALIZE KONTEKSTA ===\n")
        
        texts_to_translate = []
        translation_infos = []
        
        # Prvo obradi sve disklejmere s obzirom na kontekst
        print("\nObrada disklejmera...")
        for i, subtitle in enumerate(subtitles):
            # Dohvati kontekst za trenutni titl
            context = self._get_surrounding_context(i, subtitles)
            # Obradi disklejmere s obzirom na kontekst
            processed_text = self._process_discourse_marker(subtitle['text'], context)
            if processed_text != subtitle['text']:
                subtitle['text'] = processed_text
        
        # Zatim nastavi s normalnim prijevodom
        print("\nPokrećem batch prevođenje...")
        for i, subtitle in enumerate(subtitles):
            text = subtitle['text']
            info = subtitle.get('translation_info', {})
            
            # Spremi originalni tekst za kasniju obradu
            info['src_text'] = text
            translation_infos.append(info)
            texts_to_translate.append(self._prepare_text_for_translation(text, info))
    
        try:
            inputs = self.tokenizer(texts_to_translate, 
                                  return_tensors="pt", 
                                  padding=True, 
                                  truncation=True, 
                                  max_length=512)
            
            translations = self.model.generate(**inputs, max_length=512)
            translated_texts = [self.tokenizer.decode(t, skip_special_tokens=True) 
                              for t in translations]
            
            print("\nRezultati batch prevođenja:")
            for orig, trans in zip(texts_to_translate, translated_texts):
                print(f"  Original: {orig}")
                print(f"  Prevod:   {trans}")
                print()
            
            print("=== ZAVRŠENO BATCH PREVOĐENJE ===\n")
            for i, translation in enumerate(translated_texts):
                translation = re.sub(r'\s*<(male|female)>\s*', '', translation)
                translation = re.sub(r'\s*\[[^\]]*\]\s*', '', translation)
                translation = re.sub(r'\s+', ' ', translation)
                
                # Naknadna obrada prijevoda
                post = self._postprocess_translation(translation.strip(), translation_infos[i])
                
                # Ispiši konačni prijevod
                print(f"\n=== KONAČNI PRIJEVOD ===")
                print(f"Original: {texts_to_translate[i]}")
                print(f"Prevod:   {post}")
                
                # Analiza konteksta i provjera kvalitete prijevoda
                print("\n=== DETALJNA ANALIZA PRIJEVODA ===")
                
                # 1. Osnovne informacije
                print(f"\n[OSNOVNE INFORMACIJE]")
                print(f"Original: {texts_to_translate[i]}")
                print(f"Prevod:   {post}")
                
                # 2. Analiza konteksta
                print("\n[KONTEKST]")
                if i > 0:
                    prev_orig = texts_to_translate[i-1] if i > 0 else 'N/A'
                    prev_trans = subtitles[i-1].get('text', 'N/A') if i > 0 else 'N/A'
                    print(f"Prethodni: {prev_orig}")
                    print(f"Prethodni (prevedeno): {prev_trans}")
                
                if i < len(texts_to_translate)-1:
                    next_orig = texts_to_translate[i+1] if i < len(texts_to_translate)-1 else 'N/A'
                    print(f"Sljedeći:  {next_orig}")
                
                # 3. Provjera gramatike i pravopisa
                print("\n[GRAMATIKA I PRAVOPIS]")
                self._check_grammar_and_spelling(post, texts_to_translate[i])
                
                # 4. Provjera dosljednosti
                print("\n[DOSLJEDNOST]")
                self._check_consistency(post, prev_trans if i > 0 else None, translation_infos[i])
                
                # 5. Provjera prijevoda ključnih riječi
                print("\n[KLJUČNE RIJEČI]")
                self._check_keywords(texts_to_translate[i], post)
                
                # 6. Provjera prijevoda imenica
                print("\n[IMENICE]")
                self._check_nouns(texts_to_translate[i], post)
                
                # 7. Provjera prijevoda glagola
                print("\n[GLAGOLI]")
                self._check_verbs(texts_to_translate[i], post)
                
                print("\n=== KRAJ ANALIZE ===\n")
                
                # Gemini završna obrada (ako je uključeno)
                if getattr(self, '_gem_enabled', False):
                    if not getattr(self, '_gem_api_key', '') or not getattr(self, '_gem_model', ''):
                        # Upozori prvi put u ovoj sesiji
                        if not hasattr(self, '_gem_warned_missing'):  # one-time
                            print("⚠️  Gemini uključen, ali GOOGLE_API_KEY ili GEMINI_MODEL nisu postavljeni. Preskačem LLM poliranje.")
                            setattr(self, '_gem_warned_missing', True)
                    else:
                        # Daj Geminiju originalni polu-prevedeni izvor
                        src = translation_infos[i].get('src_text', '') 
                        neighbors = []
                        try:
                            # susjedne replike kao kontekst
                            if i > 0:
                                neighbors.append(subtitles[i-1]['text'])
                            if i+1 < len(subtitles):
                                neighbors.append(subtitles[i+1]['text'])
                        except Exception:
                            neighbors = []
                        print(f"🤖 Gemini: poliranje titla {i+1}/{len(translated_texts)}...")
                        refined = self._gemini_refine(src, post, translation_infos[i], neighbors)
                        # Low-diff kontrola: prihvati samo ako nije previše različito
                        try:
                            ratio = difflib.SequenceMatcher(a=post, b=refined or '').ratio()
                        except Exception:
                            ratio = 1.0
                        if isinstance(refined, str) and refined.strip() and ratio >= 0.6:
                            post = refined
                
                # Fallback ako rezultat izgleda nevaljan
                invalids = {'none', 'nijedan', 'nijedno', 'nijedna', 'null'}
                if not isinstance(post, str) or not post.strip() or post.strip().lower() in invalids:
                    post = translation.strip() or (translation_infos[i].get('src_text') or '')
                
                subtitles[i]['text'] = post
                
        except Exception as e:
            print(f"⚠️  Greška u prijevodu: {e}")
            for subtitle in subtitles:
                # Vrati tekst koji je ušao u batch (npr. "Pa, I don't know")
                subtitle['text'] = subtitle.get('translation_info', {}).get('src_text', subtitle['text'])

    def _check_grammar_and_spelling(self, translated_text: str, original_text: str) -> None:
        """Provjerava gramatičku točnost i pravopis prijevoda."""
        # Provjera točaka i zareza
        if translated_text and translated_text[-1] not in '.!?':
            print("  ⚠️  Nedostaje točka na kraju rečenice")
            
        # Provjera velikih slova na početku rečenice
        if translated_text and translated_text[0].islower():
            print(f"  ⚠️  Nedostaje veliko slovo na početku: {translated_text[0]}")
            
        # Provjera duplih razmaka
        if '  ' in translated_text:
            print("  ⚠️  Pronađeni dupli razmaci")
    
    def _check_consistency(self, translated_text: str, prev_translation: Optional[str], info: Dict) -> None:
        """Provjerava dosljednost u prijevodu."""
        # Provjera dosljednosti glagolskih oblika
        if prev_translation:
            prev_verbs = ['je', 'jesi', 'jeste', 'jesmo', 'jesu', 'sam', 'si', 'smo', 'ste', 'su']
            current_verbs = prev_verbs + ['biti', 'bih', 'bi', 'bismo', 'biste', 'bi']
            
            prev_has_verb = any(f' {verb} ' in f' {prev_translation.lower()} ' for verb in prev_verbs)
            curr_has_verb = any(f' {verb} ' in f' {translated_text.lower()} ' for verb in current_verbs)
            
            if prev_has_verb and curr_has_verb:
                print("  ✓ Dosljednost glagolskih oblika: DA")
            
            # Provjera dosljednosti lica
            prev_person = None
            curr_person = None
            
            # Pokušaj odrediti lice iz prethodnog prijevoda
            if 'ti ' in prev_translation.lower() or 'te ' in prev_translation.lower() or 'ti.' in prev_translation.lower():
                prev_person = '2nd'
            elif 'ja ' in prev_translation.lower() or 'mene ' in prev_translation.lower():
                prev_person = '1st'
                
            # Pokušaj odrediti lice iz trenutnog prijevoda
            if 'ti ' in translated_text.lower() or 'te ' in translated_text.lower() or 'ti.' in translated_text.lower():
                curr_person = '2nd'
            elif 'ja ' in translated_text.lower() or 'mene ' in translated_text.lower():
                curr_person = '1st'
                
            if prev_person and curr_person and prev_person != curr_person:
                print(f"  ⚠️  Nekonzistentnost lica: prethodno {prev_person}, sada {curr_person}")
    
    def _check_keywords(self, original: str, translated: str) -> None:
        """Provjerava ključne riječi u prijevodu."""
        keywords = {
            'yes': ['da', 'jesam', 'jeste', 'jesu'],
            'no': ['ne', 'nisam', 'niste', 'nisu'],
            'well': ['pa', 'dobro', 'e pa'],
            'but': ['ali', 'no', 'već'],
            'and': ['i', 'te', 'a'],
            'or': ['ili', 'ili pak']
        }
        
        original_lower = original.lower()
        translated_lower = translated.lower()
        
        for eng_word, hr_options in keywords.items():
            if eng_word in original_lower.split():
                found = any(hr_word in translated_lower for hr_word in hr_options)
                if found:
                    print(f"  ✓ Ključna riječ '{eng_word}' prevedena ispravno")
                else:
                    print(f"  ⚠️  Ključna riječ '{eng_word}' nije prevedena očekivanim prijevodom")
    
    def _check_nouns(self, original: str, translated: str) -> None:
        """Provjerava prijevod imenica."""
        # Ovdje možemo dodati specifične imenice za praćenje
        nouns = {
            'time': 'vrijeme',
            'man': 'čovjek',
            'woman': 'žena',
            'house': 'kuća',
            'car': 'auto'
        }
        
        for eng, hr in nouns.items():
            if eng in original.lower() and hr not in translated.lower():
                print(f"  ⚠️  Moguća nekonzistentnost u prijevodu imenice: {eng} -> {hr}")
    
    def _check_verbs(self, original: str, translated: str) -> None:
        """Provjerava prijevod glagola."""
        verbs = {
            'is': ['je', 'jeste'],
            'are': ['su', 'jesu', 'ste'],
            'was': ['bješe', 'bijaše', 'bilo je'],
            'were': ['bili su', 'bile su', 'bilo je']
        }
        
        original_lower = original.lower()
        translated_lower = translated.lower()
        
        for eng_verb, hr_options in verbs.items():
            if f' {eng_verb} ' in f' {original_lower} ':
                found = any(f' {hr_verb} ' in f' {translated_lower} ' for hr_verb in hr_options)
                if not found:
                    print(f"  ⚠️  Moguć problem s prijevodom glagola: {eng_verb}")
    
    def _postprocess_translation(self, text: str, info: Optional[Dict]) -> str:
        """Naknadna obrada prevedenog teksta."""
        if not text:
            return text
        gender = (info or {}).get('gender') if info else None
        src_text = (info or {}).get('src_text') if info else None # Ovo je npr. "Pa, I don't know"
        
        if gender == 'female':
            text = self._apply_feminine_corrections(text)
            
        # Primijeni lažne prijatelje (ali ne discourse markers) na HRVATSKI tekst
        # Oprez: _apply_false_friends primjenjuje SVE iz _user_pairs,
        # što uključuje i discourse markers (koji su EN regex).
        # Ovo je greška u dizajnu, ali pošto su discourse markeri
        # EN regex, neće se (uglavnom) primijeniti na HR tekst.
        text = self._apply_false_friends(text, apply_discourse_markers=False)
        
        # Heuristika za 'your friend' -> prijatelj/prijateljica prema CSV matchu teksta
        if src_text:
            # Moramo normalizirati originalni EN tekst, ne "Pa, I don't know"
            # Nemamo originalni EN tekst ovdje, što je problem.
            # Za sada koristimo src_text kakav jest.
            norm_src = self._normalize_eng(src_text) 
            if norm_src and 'your friend' in norm_src:
                speak_gender = self._csv_text_gender.get(norm_src)
                if speak_gender == 'female':
                    text = self._apply_friend_female(text)
        return text

    def _gemini_refine(self, src_text: str, draft_hr: str, info: Dict, neighbor_lines: List[str]) -> str:
        """Poziv Google Gemini API-ja za minimalno poliranje HR titla uz jasne upute."""
        try:
            if not getattr(self, '_gem_api_key', '') or not getattr(self, '_gem_model', ''):
                return draft_hr
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self._gem_model}:generateContent"
            system_prompt = (
                "Ti si lektor hrvatskog jezika za SRT titlove. Minimalno mijenjaj tekst. "
                "Očuvaj značenje, ton i stil. Poštuj rod govornika. "
                "Ne mijenjaj SRT format (ne spajaj/ne razdvajaj replike). "
                "Ako je moguće, zadrži do 2 linije i oko 42 znaka/liniji. "
                "Vrati samo konačni hrvatski tekst, bez objašnjenja."
            )
            
            # Dajemo Geminiju polu-prevedeni izvor, jer je to ono što je model preveo
            user_prompt = (
                f"Izvor (EN, ponekad polu-preveden): {src_text}\n"
                f"HR nacrt (od modela): {draft_hr}\n"
                f"Info: gender={info.get('gender')}, plural={info.get('plural_type')}, neutral_nouns={info.get('neutral_nouns')}\n"
                f"Kontekst susjednih replika: {neighbor_lines}\n"
                "Zadatak: Ako je nacrt dobar, vrati ga identično. U suprotnom, minimalno ga ispravi, poštujući rod i format."
            )
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": system_prompt + "\n\n" + user_prompt}]}
                ],
                "generationConfig": {
                    "temperature": getattr(self, '_gem_temp', 0.3),
                    "maxOutputTokens": getattr(self, '_gem_max_tokens', 120)
                }
            }
            params = {"key": self._gem_api_key}
            resp = requests.post(endpoint, params=params, json=payload, timeout=getattr(self, '_gem_timeout', 12))
            if resp.status_code != 200:
                return draft_hr
            data = resp.json() or {}
            cands = data.get("candidates") or []
            if not cands:
                return draft_hr
            parts = cands[0].get("content", {}).get("parts", [])
            text_out = None
            if parts and isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict) and p.get("text"):
                        text_out = p["text"]
                        break
            text_out = (text_out or '').strip()
            return text_out or draft_hr
        except Exception:
            return draft_hr

    def _normalize_eng(self, s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[\[\]\(\)<>]", " ", s)
        s = re.sub(r"[^a-z0-9\s']", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def _load_discourse_markers(self) -> None:
        """Učitava ugrađene prijevode ispunjavača pauza (discourse markers).
        
        Ovi se primjenjuju s visokim prioritetom kako bi osigurali
        prirodnije prijevode funkcijskih riječi u kontekstu dijaloga.
        """
        discourse_markers = [
            # Well (kontekstualno)
            (r'^\s*well\s*[,.!?]?\s*$', 'Pa', 100),            # "Well." → "Pa."
            (r'^\s*well\s*,\s*(.*)', r'Pa, \1', 100),          # "Well, ..." → "Pa, ..."
            (r'^\s*well\s+(.*)', r'Pa \1', 100),               # "Well I..." → "Pa I..."
            (r'\bwell\b\s*[,.!?]?\s*$', 'pa', 95),            # "...well." → "...pa."
            
            # So (kontekstualno)
            (r'^\s*so\s*[,.!?]?\s*$', 'Dakle', 100),           # "So." → "Dakle."
            (r'^\s*so\s*,\s*(.*)', r'Dakle, \1', 100),        # "So, ..." → "Dakle, ..."
            (r'^\s*so\s+(.*)', r'Dakle \1', 100),             # "So I..." → "Dakle I..."
            (r'\bso\b\s*[,.!?]?\s*$', 'dakle', 95),           # "...so." → "...dakle."
            
            # Now (vremenski marker u kontekstu)
            (r'^\s*now\s*,\s*(.*)', r'Sad, \1', 95),          # "Now, ..." → "Sad, ..."
            (r'^\s*now\s+then\b', 'E sad', 95),               # "Now then" → "E sad"
            (r'\bjust\s+now\b', 'upravo sada', 90),           # "just now" → "upravo sada"
            
            # Česti ispunjavači i disklejmeri
            (r'\banyway\b', 'u svakom slučaju', 90),
            (r'\bbasically\b', 'u osnovi', 90),
            (r'\bhonestly\b', 'iskreno', 90),
            (r'\bactually\b', 'zapravo', 85),
            (r'\bliterally\b', 'doslovno', 85),
            (r'\btechnically\b', 'tehnički', 85),
            (r'\bfrankly\b', 'iskreno', 90),
            (r'\bto be honest\b', 'da budem iskren', 95),
            (r'\bto tell the truth\b', 'da vam pravo kažem', 95),
            
            # Metadiskursni markeri
            (r'\bi\s+mean\b', 'mislim', 90),
            (r'\bi\s+guess\b', 'valjda', 90),
            (r'\bi\s+think\b', 'mislim', 85),
            (r'\bi\s+suppose\b', 'pretpostavljam', 85),
            (r'\bi\s+believe\b', 'vjerujem', 85),
            (r'\bi\s+feel\s+like\b', 'osjećam se kao', 85),
            (r'\bi\s+don\'t\s+know\b', 'ne znam', 90),
            (r'\bi\s+mean\s*,\s*(.*)', r'mislim, \1', 90),   # "I mean, ..." → "mislim, ..."
            
            # Interakcije s sugovornikom
            (r'\byou\s+know\b', 'znaš', 90),
            (r'\byou\s+see\b', 'vidiš', 90),
            (r'\byou\s+know\s+what\s+i\s+mean\b', 'znaš na što mislim', 95),
            (r'\bdon\'t\s+you\s+think\b', 'zar ne misliš', 90),
            (r'\bdon\'t\s+you\s+agree\b', 'zar se ne slažeš', 90),
            
            # Česte fraze za početak rečenice
            (r'^\s*look\s*,\s*(.*)', r'Gle, \1', 95),         # "Look, ..." → "Gle, ..."
            (r'^\s*listen\s*,\s*(.*)', r'Slušaj, \1', 95),    # "Listen, ..." → "Slušaj, ..."
            (r'^\s*okay\s*,\s*(.*)', r'Dobro, \1', 95),       # "Okay, ..." → "Dobro, ..."
            (r'^\s*right\s*,\s*(.*)', r'Dobro, \1', 95),      # "Right, ..." → "Dobro, ..."
            
            # Diskurzni prijelazi
            (r'\bby the way\b', 'usput rečeno', 90),
            (r'\bto be honest\b', 'da budem iskren', 95),
            (r'\bto tell you the truth\b', 'da vam pravo kažem', 95),
            (r'\bas I was saying\b', 'kako sam već rekao', 90),
            (r'\bgoing back to\b', 'da se vratim na', 90),
            (r'\bas for\b', 'što se tiče', 90),
            (r'\btalking about\b', 'kad smo već kod', 90),
            
            # Česti izrazi sa 'kind of' i 'sort of'
            (r'\bkind of\b', 'nekako', 90),
            (r'\bsort of\b', 'pomalo', 90),
            (r'\bkind of like\b', 'nešto poput', 90),
            
            # Česte fraze s 'like'
            (r'\bit\'s\s+like\b', 'to je kao', 85),
            (r'\bi\'m\s+like\b', 'ja sam ono', 85),
            (r'\bhe\'s\s+like\b', 'on je ono', 85),
            (r'\bshe\'s\s+like\b', 'ona je ono', 85),
            (r'\bthey\'re\s+like\b', 'oni su ono', 85),
            
            # Česti izrazi s 'you know'
            (r'\byou know what\b', 'znaš što', 90),
            (r'\byou know what I mean\b', 'znaš na što mislim', 95),
            (r'\byou know what I\'m saying\b', 'znaš na što mislim', 95),
            
            # Česti izrazi s 'I mean'
            (r'\bi mean\b', 'mislim', 90),
            (r'\bi mean\s*,\s*(.*)', r'mislim, \1', 90),
            
            # Česti izrazi s 'I guess'
            (r'\bi guess\b', 'valjda', 90),
            (r'\bi guess\s*,\s*(.*)', r'valjda, \1', 90),
            
            # Česti izrazi s 'I think'
            (r'\bi think\b', 'mislim', 85),
            (r'\bi think\s*,\s*(.*)', r'mislim, \1', 85),
            
            # Česti izrazi s 'I suppose'
            (r'\bi suppose\b', 'pretpostavljam', 85),
            (r'\bi suppose\s*,\s*(.*)', r'pretpostavljam, \1', 85),
            
            # Česti izrazi s 'I believe'
            (r'\bi believe\b', 'vjerujem', 85),
            (r'\bi believe\s*,\s*(.*)', r'vjerujem, \1', 85),
            
            # Česti izrazi s 'I feel'
            (r'\bi feel\b', 'osjećam', 85),
            (r'\bi feel\s+like\b', 'osjećam se kao', 85),
            (r'\bi feel\s*,\s*(.*)', r'osjećam, \1', 85),
            
            # Česti izrazi s 'I don\'t know'
            (r'\bi don\'t know\b', 'ne znam', 90),
            (r'\bi don\'t know\s*,\s*(.*)', r'ne znam, \1', 90),
            
            # Česti izrazi s 'you know'
            (r'\byou know\b', 'znaš', 90),
            (r'\byou know\s*,\s*(.*)', r'znaš, \1', 90),
            
            # Česti izrazi s 'you see'
            (r'\byou see\b', 'vidiš', 90),
            (r'\byou see\s*,\s*(.*)', r'vidiš, \1', 90),
            
            # Česti izrazi s 'look'
            (r'^\s*look\b', 'Gle', 95),
            (r'^\s*look\s*,\s*(.*)', r'Gle, \1', 95),
            
            # Česti izrazi s 'listen'
            (r'^\s*listen\b', 'Slušaj', 95),
            (r'^\s*listen\s*,\s*(.*)', r'Slušaj, \1', 95),
            
            # Česti izrazi s 'okay' i 'ok'
            (r'^\s*okay\b', 'Dobro', 95),
            (r'^\s*okay\s*,\s*(.*)', r'Dobro, \1', 95),
            (r'^\s*ok\b', 'Dobro', 95),
            (r'^\s*ok\s*,\s*(.*)', r'Dobro, \1', 95),
            
            # Česti izrazi s 'right'
            (r'^\s*right\b', 'Dobro', 95),
            (r'^\s*right\s*,\s*(.*)', r'Dobro, \1', 95),
            
            # Česti izrazi s 'well then'
            (r'^\s*well then\b', 'Pa dobro', 95),
            (r'^\s*well then\s*,\s*(.*)', r'Pa dobro, \1', 95),
            
            # Česti izrazi s 'so then'
            (r'^\s*so then\b', 'Dakle', 95),
            (r'^\s*so then\s*,\s*(.*)', r'Dakle, \1', 95),
            
            # Česti izrazi s 'now then'
            (r'^\s*now then\b', 'E sad', 95),
            (r'^\s*now then\s*,\s*(.*)', r'E sad, \1', 95),
            
            # Česti izrazi s 'anyway'
            (r'^\s*anyway\b', 'U svakom slučaju', 95),
            (r'^\s*anyway\s*,\s*(.*)', r'U svakom slučaju, \1', 95),
            
            # Česti izrazi s 'anyhow'
            (r'^\s*anyhow\b', 'Kako god', 95),
            (r'^\s*anyhow\s*,\s*(.*)', r'Kako god, \1', 95),
            
            # Česti izrazi s 'anyways'
            (r'^\s*anyways\b', 'U svakom slučaju', 95),
            (r'^\s*anyways\s*,\s*(.*)', r'U svakom slučaju, \1', 95),
            
            # Česti izrazi s 'so yeah'
            (r'^\s*so yeah\b', 'Pa da', 95),
            (r'^\s*so yeah\s*,\s*(.*)', r'Pa da, \1', 95),
            
            # Česti izrazi s 'so um'
            (r'^\s*so um\b', 'Pa ovaj', 95),
            (r'^\s*so um\s*,\s*(.*)', r'Pa ovaj, \1', 95),
            
            # Česti izrazi s 'so uh'
            (r'^\s*so uh\b', 'Pa ovaj', 95),
            (r'^\s*so uh\s*,\s*(.*)', r'Pa ovaj, \1', 95),
        ]
        
        # Sortiranje po prioritetu (od najvišeg prema najnižem)
        discourse_markers.sort(key=lambda x: -x[2])
        
        # Dodavanje u self._user_pairs s kompiliranim regex uzorcima
        for pattern, replacement, priority in discourse_markers:
            self._user_pairs.append((
                re.compile(pattern, re.IGNORECASE),
                replacement,
                priority
            ))

    def _get_surrounding_context(self, current_idx: int, all_subtitles: List[dict], window: int = 2) -> Dict[str, List[str]]:
        """Dohvaća kontekst oko trenutnog titla.
        
        Args:
            current_idx: Indeks trenutnog titla
            all_subtitles: Lista svih titlova
            window: Koliko titlova prije i poslije uzeti u obzir
            
        Returns:
            Rječnik s ključevima 'prev_texts' i 'next_texts' koji sadrže listu prethodnih i sljedećih tekstova
        """
        context = {
            'prev_texts': [],
            'next_texts': []
        }
        
        # Dohvati prethodne titlove
        for i in range(max(0, current_idx - window), current_idx):
            if i >= 0 and i < len(all_subtitles):
                context['prev_texts'].append(all_subtitles[i]['text'])
        
        # Dohvati sljedeće titlove
        for i in range(current_idx + 1, min(current_idx + window + 1, len(all_subtitles))):
            if i < len(all_subtitles):
                context['next_texts'].append(all_subtitles[i]['text'])
        
        return context

    def _process_discourse_marker(self, text: str, context: Dict) -> str:
        """Obrada ispunjavača pauze s obzirom na kontekst.
        
        Poseban slučaj za "Well..." -> "Pa..."
        
        Args:
            text: Tekst za obradu
            context: Rječnik s kontekstom (prev_texts, next_texts)
            
        Returns:
            Obradjeni tekst s primijenjenim zamjenama ispunjavača pauza
        """
        if not text.strip():
            return text
            
        original_text = text
        normalized = text.lower().strip()
        
        # Dohvati susjedne tekstove iz konteksta
        prev_texts = context.get('prev_texts', [])
        next_texts = context.get('next_texts', [])
        
        # Koristimo samo najbliže susjede za analizu konteksta
        prev_line = prev_texts[-1] if prev_texts else ""
        next_line = next_texts[0] if next_texts else ""
        
        # Provjeri je li riječ o ispunjavaču pauze
        for pattern, replacement, _ in self._user_pairs:
            if pattern.search(normalized):
                # Ispiši informacije o pronađenom ispunjavaču
                print(f"\n🔍 Pronađen ispunjavač u tekstu: '{text}'")
                print(f"   • Uzorak: {pattern.pattern}")
                print(f"   • Zamjena: '{replacement}'")
                print(f"   • Kontekst: {'...' if prev_line else 'N/A'} | {text} | {'...' if next_line else ''}")
                
                # Analiza konteksta
                if prev_line:
                    text_lower = text.strip().lower()
                    if text_lower in ["yes", "yes."]:
                        print(f"Pronađen kratak odgovor '{text}' - prevodim u 'Da'")
                        return "Da"
                    elif text_lower in ["no", "no."]:
                        print(f"Pronađen kratak odgovor '{text}' - prevodim u 'Ne'")
                        return "Ne"
                        
                    # Provjeri je li tekst samo "Well..."
                    if text_lower in ["well...", "well"]:
                        print(f"Pronađen disklejmer 'Well...' - mijenjam u 'Pa...'")
                        return "Pa..."
                        
                    # Provjeri "Well..." na početku rečenice
                    if text.strip().lower().startswith("well..."):
                        print(f"Pronađen 'Well...' na početku rečenice - mijenjam u 'Pa...'")
                        return "Pa..." + text[6:]  # Zadrži ostatak teksta nakon 'Well...'
                        
                    # Provjeri "Well, " na početku rečenice
                    if text.strip().lower().startswith("well, "):
                        print(f"Pronađen 'Well, ' na početku rečenice - mijenjam u 'Pa, '")
                        return "Pa, " + text[6:]  # Zadrži ostatak teksta nakon 'Well, '
                        
                    prev = prev_line.lower()
                    next_line_lower = next_line.lower()
                    
                    # Provjera upitnika u prethodnoj rečenici
                    if prev.rstrip().endswith('?'):
                        print(f"   ✓ Zadržavam ispunjavač jer prethodna rečenica završava upitnikom")
                        return text
                        
                    # Provjera pitanja u sljedećoj rečenici
                    question_words = ('zašto', 'kako', 'što', 'jesu li', 'ima li', 'da li', 'hoćeš li', 'jeste li')
                    if any(next_line_lower.strip().startswith(word) for word in question_words):
                        print(f"   ✓ Zadržavam ispunjavač jer slijedi pitanje")
                        return text
                
                # Primjena zamjene
                new_text = pattern.sub(replacement, text, count=1).strip()
                if new_text != text:
                    print(f"   • Primjenjujem zamjenu: '{text}' -> '{new_text}'")
                    return new_text
                
                return text
        
        return text

    def _apply_feminine_corrections(self, text: str) -> str:
        rules = [
            (r"\brazmišljao\b", "razmišljala"),
            (r"\bmisli(o)?\b", "mislila"),
            (r"\bmogao\b", "mogla"),
            (r"\bnije mogao\b", "nije mogla"),
            (r"\bšokiran\b", "šokirana"),
            (r"\bhtio\b", "htjela"),
            (r"\bučinio\b", "učinila"),
            (r"\bnapravio\b", "napravila"),
            (r"\brekao\b", "rekla"),
        ]
        def replace_case(m, repl):
            s = m.group(0)
            return repl.capitalize() if s[:1].isupper() else repl
        for pat, repl in rules:
            text = re.sub(pat, lambda m: replace_case(m, repl), text, flags=re.IGNORECASE)
        return text

    def _apply_false_friends(self, text: str, apply_discourse_markers: bool = True) -> str:
        """Primjenjuje ispravke za lažne prijatelje i kontekstualne razlike.
        
        Redoslijed primjene:
        1. Ugrađeni lažni prijatelji (nizak prioritet)
        2. Korisnički parovi + discourse markers (sortirani po prioritetu)
        
        Argument 'apply_discourse_markers' je dodan da se može kontrolirati
        da li se discourse markeri (koji su EN->HR) primjenjuju
        na tekst (korisno kod post-procesiranja HR teksta).
        """
        if not text:
            return text
            
        # Kopiraj tekst za obradu
        processed = text
        
        # Prvo obradi sve kontekstualne zamjene
        for pattern, replacement, _ in sorted(self._user_pairs, key=lambda x: -x[2]):
            if not apply_discourse_markers and 'discourse_marker' in str(pattern):
                continue
                
            # Provjeri je li ovo kontekstualna zamjena
            if '[u kontekstu:' in str(pattern):
                # Izvuci ključ i kontekst
                match = re.search(r'(.+?)\s*\[u kontekstu:(.+?)\]', str(pattern.pattern))
                if match:
                    key = match.group(1).strip()
                    context = match.group(2).strip().lower()
                    
                    # Provjeri postoji li kontekst u tekstu
                    if context in processed.lower():
                        # Koristimo funkciju za zamjenu koja čuva velika i mala slova
                        def repl(match):
                            s = match.group(0)
                            return replacement.capitalize() if s and s[0].isupper() else replacement
                            
                        # Primijeni zamjenu samo ako se ključ pojavljuje u tekstu
                        if key.lower() in processed.lower():
                            processed = re.sub(
                                re.escape(key), 
                                lambda m: replacement.capitalize() if m.group(0)[0].isupper() else replacement,
                                processed,
                                flags=re.IGNORECASE
                            )
        
        # Zatim obiđi sve ostale zamjene
        for pattern, replacement, _ in sorted(self._user_pairs, key=lambda x: -x[2]):
            if not apply_discourse_markers and 'discourse_marker' in str(pattern):
                continue
                
            # Preskoči već obrađene kontekstualne zamjene
            if '[u kontekstu:' in str(pattern):
                continue
                
            # Koristimo funkciju za zamjenu koja čuva velika i mala slova
            def repl(match):
                s = match.group(0)
                return replacement.capitalize() if s and s[0].isupper() else replacement
                
            # Primijeni zamjenu
            processed = pattern.sub(repl, processed)
        
        return processed if processed != text else text

    def _apply_friend_female(self, text: str) -> str:
        # Zamjene prilagodbe za prijatelj -> prijateljica (najčešći padeži)
        pairs = [
            (r"\brazmišljao\b", "razmišljala"),
            (r"\bmisli(o)?\b", "mislila"),
            (r"\bmogao\b", "mogla"),
            (r"\bnije mogao\b", "nije mogla"),
            (r"\bšokiran\b", "šokirana"),
            (r"\bhtio\b", "htjela"),
            (r"\bučinio\b", "učinila"),
            (r"\bnapravio\b", "napravila"),
            (r"\brekao\b", "rekla"),
        ]
        def replace_case(m, repl):
            s = m.group(0)
            return repl.capitalize() if s[:1].isupper() else repl
        for pat, repl in pairs:
            text = re.sub(pat, lambda m: replace_case(m, repl), text, flags=re.IGNORECASE)
        return text

    def _load_user_dictionary(self, path: str) -> None:
        """Učitava korisnički rječnik iz CSV ili TXT datoteke.
        
        Podržani formati:
        - CSV s kolonama: key,replacement,priority
        - TXT s linijama u formatu: kljuc -> vrijednost
        - Podržava kontekst u uglatim zagradama: kljuc [u kontekstu:xyz] -> vrijednost
        """
        pairs: List[Tuple[str, str, int]] = []
        
        if path.lower().endswith('.csv'):
            with open(path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get('key') or row.get('src') or '').strip()
                    repl = (row.get('replacement') or row.get('dst') or '').strip()
                    prio = row.get('priority')
                    try:
                        pr = int(prio) if prio is not None and str(prio).strip() != '' else 0
                    except ValueError:
                        pr = 0
                    if key and repl:
                        pairs.append((key, repl, pr))
        else:
            # TXT: podržava -> i = kao razdjelnike, te kontekst u uglatim zagradama
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    # Podrži i -> i = kao razdjelnike
                    if '->' in line:
                        key, repl = line.split('->', 1)
                    elif '=' in line:
                        key, repl = line.split('=', 1)
                    else:
                        continue
                        
                    key = key.strip()
                    repl = repl.strip()
                    
                    # Dodaj par u listu s prioritetom 0 (može se nadjačati kasnije)
                    if key and repl:
                        pairs.append((key, repl, 0))
        
        # Kompiliraj nove parove
        compiled: List[Tuple[re.Pattern, str, int]] = []
        for key, repl, pr in pairs:
            escaped = re.escape(key)
            if re.match(r'^[\w\s]+$', key, flags=re.UNICODE) and '\\s' not in key:
                pattern = re.compile(rf"\b{escaped}\b", flags=re.IGNORECASE)
            else:
                pattern = re.compile(escaped, flags=re.IGNORECASE)
            compiled.append((pattern, repl, pr))
        
        # ✅ NOVO: DODAJ na postojeće discourse markers, ne prebrisuj ih
        self._user_pairs.extend(compiled)

    def _find_matching_video_file(self, srt_path: str, search_dirs: Optional[List[Path]] = None) -> Optional[Path]:
        """
        Pronalazi odgovarajuću video datoteku.
        
        Args:
            srt_path: Putanja do SRT datoteke
            search_dirs: Lista direktorija za pretraživanje (ako je None, koristi se direktorij SRT-a)
        """
        srt_path = Path(srt_path)
        srt_stem = srt_path.stem
        
        # Ako nisu navedeni direktoriji za pretraživanje, koristi direktorij SRT-a
        if search_dirs is None:
            search_dirs = [srt_path.parent]
        elif not isinstance(search_dirs, list):
            search_dirs = [search_dirs]
        
        # Dodaj i direktorij gdje se spremaju prevedeni titlovi (ako je različit)
        output_dir = srt_path.parent / "Prevedeno"
        if output_dir.exists() and output_dir not in search_dirs:
            search_dirs.append(output_dir)
        
        # Dodaj i roditeljski direktorij (za slučaj da su video datoteke u njemu)
        parent_dir = srt_path.parent.parent
        if parent_dir.exists() and parent_dir not in search_dirs:
            search_dirs.append(parent_dir)
        
        # Ukloni ekstenziju i jezične oznake
        base_name = re.sub(r'[._-]*(?:en|hr|eng|hrv|srt|sub|dvdrip|webrip|hdtv)[._-]*', '', srt_stem, flags=re.IGNORECASE)
        base_name = re.sub(r'[._-]+', ' ', base_name).strip()
        
        # Podržani video formati
        video_extensions = ['.mkv', '.mp4', '.avi', '.mov', '.wmv', '.m4v']
        
        # Prvo pokušaj pronaći točno podudaranje s baznim imenom
        for directory in search_dirs:
            if not directory.exists():
                continue
                
            for ext in video_extensions:
                for video_file in directory.glob(f"*{ext}"):
                    video_stem = video_file.stem
                    
                    # Ukloni kvalitete i formate iz imena
                    clean_video_stem = re.sub(
                        r'[._-]*(?:1080p|720p|480p|hdtv|web|bluray|dvdrip|yts|yify|hevc|x264|x265|aac|ac3|dd5.1|dts|hvec|h.264|h.265)[._-]*', 
                        ' ', 
                        video_stem, 
                        flags=re.IGNORECASE
                    )
                    clean_video_stem = re.sub(r'[._-]+', ' ', clean_video_stem).strip()
                    
                    # Normaliziraj imena serija (S02E03 -> 2x03)
                    srt_episode = self._normalize_episode_format(base_name)
                    video_episode = self._normalize_episode_format(clean_video_stem)
                    
                    # Ako su oba imena prazna nakon normalizacije, koristi originalna imena
                    if not srt_episode and not video_episode:
                        srt_episode = base_name
                        video_episode = clean_video_stem
                    
                    # Ako su epizode iste, vrati ovu datoteku
                    if srt_episode and video_episode and srt_episode == video_episode:
                        return video_file
                    
                    # Ako nema epizoda, usporedi samo imena serija
                    if not srt_episode and not video_episode and self._are_names_similar(base_name, clean_video_stem):
                        return video_file
        
        # Ako nema točnog podudaranja, pokušaj pronaći bilo koju video datoteku s istim brojevima sezone/epizode
        srt_season_episode = self._extract_season_episode(srt_stem)
        if srt_season_episode:
            season, episode = srt_season_episode
            for directory in search_dirs:
                for ext in video_extensions:
                    for video_file in directory.glob(f"*{ext}"):
                        video_season_episode = self._extract_season_episode(video_file.stem)
                        if video_season_episode and video_season_episode == (season, episode):
                            return video_file
        
        # Ako i dalje nema podudaranja, vrati prvu pronađenu video datoteku
        for directory in search_dirs:
            for ext in video_extensions:
                video_files = list(directory.glob(f"*{ext}"))
                if video_files:
                    return video_files[0]
        
        return None
    
    def _normalize_episode_format(self, text: str) -> str:
        """Normalizira format epizode (npr. S02E03 -> 2x03)."""
        # Ukloni sve što nije broj ili x/X
        clean_text = re.sub(r'[^0-9xX]', ' ', text)
        # Pronađi sve brojeve i x/X između njih
        matches = re.findall(r'(\d+)[xX](\d+)', clean_text)
        if matches:
            season, episode = matches[0]
            # Ukloni vodeće nule
            season = str(int(season))
            episode = str(int(episode))
            return f"{season}x{episode}"
        return ""
    
    def _extract_season_episode(self, text: str) -> Optional[tuple[int, int]]:
        """Izvuci broj sezone i epizode iz teksta."""
        # Pokušaj s formatom S02E03
        match = re.search(r'[sS](\d+)[eE](\d+)', text)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        
        # Pokušaj s formatom 2x03
        match = re.search(r'(\d+)[xX](\d+)', text)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        
        # Pokušaj s formatom 2.03
        match = re.search(r'(\d+)\.(\d{2,})', text)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        
        return None

    def _are_names_similar(self, name1: str, name2: str, threshold: float = 0.7) -> bool:
        """Provjerava jesu li dva imena slična koristeći fuzzy usporedbu."""
        # Ukloni sve što nije slovo ili broj i pretvori u mala slova
        clean1 = re.sub(r'[^a-z0-9]', '', name1.lower())
        clean2 = re.sub(r'[^a-z0-9]', '', name2.lower())
        
        # Ako je jedno ime podskup drugog, smatramo da su slični
        if clean1 in clean2 or clean2 in clean1:
            return True
        
        # Inače koristimo fuzzy usporedbu
        similarity = SequenceMatcher(None, clean1, clean2).ratio()
        return similarity >= threshold

    def _generate_output_filename(self, video_path: Path, srt_path: Path) -> Path:
        """Generira izlazno ime datoteke na temelju video datoteke."""
        video_stem = video_path.stem
        return srt_path.parent / f"{video_stem}.hr.srt"

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
            print(f"❌ Greška: Ne mogu pročitati file {file_path}")
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

    def _write_srt(self, subtitles: List[dict], output_path: str):
        """Zapisuje titlove u SRT format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, sub in enumerate(subtitles):
                f.write(f"{sub['index']}\n")
                f.write(f"{sub['timestamp']}\n")
                f.write(f"{sub['text']}\n")
                if i < len(subtitles) - 1:
                    f.write("\n")

class ToolTip:
    """Klasa za kreiranje tooltip-ova."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        """Prikazuje tooltip pored widgeta."""
        if not self.tooltip:
            x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 25
            
            self.tooltip = tk.Toplevel(self.widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")
            
            label = ttk.Label(self.tooltip, text=self.text, justify='left',
                            background='#ffffe0', relief='solid', borderwidth=1,
                            padding=(5, 3, 5, 3))
            label.pack()
    
    def hide_tooltip(self, event=None):
        """Sakriva tooltip."""
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class SubtitleTranslatorApp:
    """Glavna klasa za korisničko sučelje prevoditelja titlova."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sinkronizator titlova")
        self.root.geometry("800x550")
        self.root.minsize(750, 500)
        
        # Varijable za praćenje vremena i napretka
        self.start_time = None
        self.last_update_time = None
        self.last_progress = 0
        self.estimated_total_time = 0
        self.current_file_index = 0
        self.total_files = 1
        
        # Postavi ikonu aplikacije ako postoji
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
            
        # Kreiraj izbornik
        self.create_menu()
        
        # Stilizacija
        self.setup_styles()
        
        # Glavni okvir s paddingom
        main_frame = ttk.Frame(root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Konfiguracija težina redova i stupaca
        main_frame.columnconfigure(1, weight=1)
        
        # Naslov
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 15))
        ttk.Label(title_frame, text="Sinkronizator titlova", 
                 font=('Segoe UI', 14, 'bold')).pack()
        
        # Okvir s postavkama
        settings_frame = ttk.LabelFrame(main_frame, text=" Postavke ", padding=10)
        settings_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=5)
        
        # Red s opcijama
        options_frame = ttk.Frame(settings_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        # Batch mod
        self.batch_mode = tk.BooleanVar(value=False)
        self.batch_cb = ttk.Checkbutton(
            options_frame, 
            text="Batch mod (obradi sve .srt u mapi)",
            variable=self.batch_mode,
            command=self.toggle_batch_mode
        )
        self.batch_cb.pack(side=tk.LEFT, padx=5)
        
        # Gemini opcija
        self.gemini_enabled = tk.BooleanVar(value=False)
        self.gemini_cb = ttk.Checkbutton(
            options_frame, 
            text="Koristi Gemini za poboljšanje prijevoda",
            variable=self.gemini_enabled
        )
        self.gemini_cb.pack(side=tk.LEFT, padx=5)
        
        # Okvir za ulaz/izlaz
        io_frame = ttk.LabelFrame(main_frame, text=" Ulaz i izlaz ", padding=10)
        io_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=5)
        
        # Funkcija za kreiranje reda s oznakom i poljem
        def create_file_row(parent, label_text, var, browse_cmd, row, is_folder=False):
            ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=3, padx=5)
            entry = ttk.Entry(parent, textvariable=var, width=60)
            entry.grid(row=row, column=1, padx=5, pady=3, sticky=tk.EW)
            btn_text = "Odaberi mapu..." if is_folder else "Odaberi datoteku..."
            btn = ttk.Button(parent, text=btn_text, command=browse_cmd)
            btn.grid(row=row, column=2, padx=5, pady=3)
            parent.columnconfigure(1, weight=1)
            return entry, btn
        
        # Ulaz
        self.input_path = tk.StringVar()
        input_entry, self.input_btn = create_file_row(
            io_frame, 
            "Ulazna datoteka/mapa:", 
            self.input_path, 
            self.browse_input, 
            0, 
            self.batch_mode.get()
        )
        
        # Izlaz
        self.output_path = tk.StringVar()
        output_entry, self.output_btn = create_file_row(
            io_frame,
            "Izlazna datoteka/mapa:",
            self.output_path,
            self.browse_output,
            1,
            self.batch_mode.get()
        )
        
        # Dodatne opcije (sakrivene po defaultu)
        self.advanced_frame = ttk.LabelFrame(main_frame, text=" Dodatne opcije ", padding=10)
        
        # Gumb za prikaz/skrivanje dodatnih opcija
        self.show_advanced = False
        self.advanced_btn = ttk.Button(
            main_frame, 
            text="▼ Dodatne opcije", 
            command=self.toggle_advanced,
            style='TButton'
        )
        self.advanced_btn.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # Dodatne opcije
        self.metadata_csv_path = tk.StringVar()
        self.user_dict_path = tk.StringVar()
        
        # Unutarnji okvir za dodatne opcije
        inner_adv_frame = ttk.Frame(self.advanced_frame)
        inner_adv_frame.pack(fill=tk.X, expand=True)
        
        create_file_row(
            inner_adv_frame,
            "CSV metapodaci:",
            self.metadata_csv_path,
            self.browse_csv,
            0
        )
        
        create_file_row(
            inner_adv_frame,
            "Korisnički rječnik:",
            self.user_dict_path,
            self.browse_user_dict,
            1
        )
        
        # Statusna traka i progress bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=10, column=0, columnspan=3, sticky='ew', pady=(15, 5))
        
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(
            status_frame, 
            variable=self.progress_var, 
            mode='determinate',
            length=100
        )
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.status_var = tk.StringVar(value="Spreman za prevođenje")
        status_label = ttk.Label(
            status_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W, 
            padding=5,
            style='Status.TLabel'
        )
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Gumbi
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=9, column=0, columnspan=3, pady=10, sticky='e')
        
        self.translate_btn = ttk.Button(
            btn_frame, 
            text="Prevedi titlove", 
            command=self.start_translation,
            style='Accent.TButton'
        )
        self.translate_btn.pack(side=tk.RIGHT, padx=5)
        
        # Tooltip-ovi
        self.tooltips = {
            'batch_mode': "Uključuje obradu svih .srt datoteka u odabranoj mapi",
            'gemini_enabled': "Koristi Google Gemini za dodatno poboljšanje prijevoda (zahtijeva internetsku vezu)",
            'input_path': "Odaberite pojedinačnu .srt datoteku ili mapu s datotekama",
            'output_path': "Odredite izlaznu datoteku ili mapu za prevedene titlove",
            'metadata_csv': "CSV datoteka s dodatnim metapodacima (npr. imena likova i rodovi)",
            'user_dict': "Korisnički rječnik s prijevodima specifičnih izraza"
        }
        
        # Povezivanje tooltip-ova
        ToolTip(self.batch_cb, self.tooltips['batch_mode'])
        ToolTip(self.gemini_cb, self.tooltips['gemini_enabled'])
        ToolTip(input_entry, self.tooltips['input_path'])
        ToolTip(output_entry, self.tooltips['output_path'])
        
        # Inicijalno sakrij napredne opcije
        self.toggle_advanced(show=False)
        
        # Povećanje prioriteta prozora
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
    
    def create_menu(self):
        """Kreira glavni izbornik aplikacije."""
        menubar = tk.Menu(self.root)
        
        # Izbornik Postavke
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Postavke API-ja", command=self.show_api_settings)
        menubar.add_cascade(label="Postavke", menu=settings_menu)
        
        # Izbornik Pomoć
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Kako koristiti", command=self.show_help)
        help_menu.add_command(label="O programu", command=self.show_about)
        menubar.add_cascade(label="Pomoć", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def show_api_settings(self):
        """Prikazuje prozor za postavke API-ja."""
        api_window = tk.Toplevel(self.root)
        api_window.title("Postavke API-ja")
        api_window.geometry("500x250")
        api_window.resizable(False, False)
        
        # Centriranje prozora
        window_width = 500
        window_height = 250
        screen_width = api_window.winfo_screenwidth()
        screen_height = api_window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        api_window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Okvir za unose
        frame = ttk.Frame(api_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Google API Key
        ttk.Label(frame, text="Google API ključ:").grid(row=0, column=0, sticky=tk.W, pady=5)
        google_key_var = tk.StringVar(value=os.getenv('GOOGLE_API_KEY', ''))
        google_entry = ttk.Entry(frame, textvariable=google_key_var, width=50, show="*")
        google_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Hugging Face API Key
        ttk.Label(frame, text="Hugging Face token:").grid(row=1, column=0, sticky=tk.W, pady=5)
        hf_key_var = tk.StringVar(value=os.getenv('HF_API_TOKEN', ''))
        hf_entry = ttk.Entry(frame, textvariable=hf_key_var, width=50, show="*")
        hf_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Gumbi
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=15)
        
        def save_api_keys():
            """Sprema unesene API ključeve u .env datoteku."""
            env_path = os.path.join(os.getcwd(), '.env')
            env_lines = []
            
            # Pročitaj postojeće postavke ako datoteka postoji
            if os.path.exists(env_path):
                with open(env_path, 'r', encoding='utf-8') as f:
                    env_lines = f.readlines()
            
            # Ažuriraj ili dodaj API ključeve
            keys_to_update = {
                'GOOGLE_API_KEY': google_key_var.get(),
                'HF_API_TOKEN': hf_key_var.get()
            }
            
            for key, value in keys_to_update.items():
                key_found = False
                key_line = f"{key}={value}\n"
                
                for i, line in enumerate(env_lines):
                    if line.startswith(f"{key}="):
                        env_lines[i] = key_line
                        key_found = True
                        break
                
                if not key_found and value:
                    env_lines.append(key_line)
            
            # Spremi sve promjene
            with open(env_path, 'w', encoding='utf-8') as f:
                f.writelines(env_lines)
            
            messagebox.showinfo("Uspjeh", "Postavke API-ja su uspješno spremljene.")
            api_window.destroy()
        
        ttk.Button(btn_frame, text="Spremi", command=save_api_keys).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Odustani", command=api_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Postavi težine redova i stupaca za pravilno poravnanje
        frame.columnconfigure(1, weight=1)
        for i in range(3):
            frame.rowconfigure(i, weight=1)
    
    def show_help(self):
        """Prikazuje prozor s uputama za korištenje."""
        help_text = """KAKO KORISTITI APLIKACIJU

1. Odaberite ulaznu datoteku (.srt) ili mapu (u batch modu)
2. Odaberite izlaznu datoteku ili mapu
3. Podesite dodatne opcije po potrebi
4. Pritisnite gumb 'Prevedi' za početak

BATCH MOD:
- Omogućuje obradu svih .srt datoteka u odabranoj mapi
- Rezultati će biti spremljeni u odabranu izlaznu mapu
- Nazivi izlaznih datoteka će biti isti kao ulazne, s .hr.srt nastavkom

DODATNE OPCIJE:
- Koristi Gemini: Koristi Google Gemini API za poboljšanje prijevoda
- Korisnički rječnik: Dodajte vlastite prijevode u datoteku osnovni_rijecnik.txt

KAKO DOBITI API KLJUČEVE:

1. Google API ključ:
   - Idite na Google Cloud Console (https://console.cloud.google.com/)
   - Kreirajte novi projekt ili odaberite postojeći
   - Omogućite "Cloud Translation API"
   - U izborniku idite na "Credentials" i kreirajte novi API ključ
   - Kopirajte ključ i zalijepite u postavkama

2. Hugging Face model i token:
   - Prijavite se na https://huggingface.co/
   - Idite na Settings > Access Tokens
   - Kliknite "New token" i slijedite upute
   - Kopirajte token i zalijepite u postavkama
   - Ako model nije lokalno dostupan, aplikacija će ga automatski preuzeti

KORIŠTENJE HUGGING FACE MODELA:

1. Prvo korištenje:
   - Prilikom prvog pokretanja, aplikacija će preuzeti potrebne modele
   - Preuzimanje može potrajati nekoliko minuta, ovisno o brzini veze
   - Modeli se spremaju lokalno za buduću upotrebu

2. Preporučeni modeli:
   - Glavni model: Helsinki-NLP/opus-mt-en-zls
   - Alternativni modeli:
     - Helsinki-NLP/opus-mt-en-sla (za slavenske jezike)
     - facebook/nllb-200-distilled-600M (za višejezično prevođenje)

3. Rješavanje problema:
   - Ako se model ne preuzima, provjerite internetsku vezu
   - Za ručno preuzimanje modela:
     1. Instalirajte transformers: pip install transformers
     2. Pokrenite Python konzolu i izvršite:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_name = "Helsinki-NLP/opus-mt-en-zls"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

ZA POMOĆ:
- Za pitanja ili probleme, posjetite nas na Facebooku:
  https://www.facebook.com/sdenis.vr/

Aplikacija razvijena uz ljubav od strane Denisa Sakača
"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Pomoć")
        help_window.geometry("600x500")
        
        # Centriranje prozora
        window_width = 600
        window_height = 500
        screen_width = help_window.winfo_screenwidth()
        screen_height = help_window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        help_window.geometry(f'{window_width}x{window_height}+{x}+{y}')
        
        # Okvir s tekstom i scrollbar-om
        frame = ttk.Frame(help_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text = tk.Text(frame, wrap=tk.WORD, font=('Segoe UI', 10), padx=10, pady=10)
        text.insert(tk.END, help_text)
        text.config(state=tk.DISABLED)
        
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(fill=tk.BOTH, expand=True)
        
        # Gumb za zatvaranje
        btn_frame = ttk.Frame(help_window)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="Zatvori", command=help_window.destroy).pack(side=tk.RIGHT)
    
    def show_about(self):
        """Prikazuje prozor s informacijama o programu."""
        about_text = """Sinkronizator titlova

Verzija: 0.12

Aplikacija za prevođenje i sinkronizaciju titlova s engleskog na hrvatski jezik.

Autor: Denis Sakač
Kontakt: https://www.facebook.com/sdenis.vr/

Koristi napredne tehnike obrade prirodnog jezika za preciznije prevođenje.

© 2023-2024 Sva prava pridržana
"""
        messagebox.showinfo("O programu", about_text)
    
    def setup_styles(self):
        """Postavlja stilove za korisničko sučelje."""
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TProgressbar', thickness=20, background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', padding=2)
        style.configure('TButton', padding=5)
        style.configure('TEntry', padding=3)
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', background='#f0f0f0', font=('Segoe UI', 9, 'bold'))
        
        # Akcentni gumb
        style.configure('Accent.TButton', font=('Segoe UI', 10, 'bold'))
        
        # Statusna traka
        style.configure('Status.TLabel', background='#e0e0e0', foreground='#333333')
        
        # Podešavanje boja za tamnu temu
        if self.root.tk.call('tk', 'windowingsystem') == 'win32':
            try:
                from ctypes import windll, byref, sizeof, c_int
                # Provjeri je li tamna tema aktivna na Windows 10/11
                if hasattr(windll, 'dwmapi'):
                    value = c_int(0)
                    windll.dwmapi.DwmGetColorizationColor(byref(value), None)
                    if value.value != 0:
                        # Tamna tema je aktivna
                        style.configure('.', background='#2b2b2b', foreground='#ffffff')
                        style.configure('TFrame', background='#2b2b2b')
                        style.configure('TLabel', background='#2b2b2b', foreground='#ffffff')
                        style.configure('TLabelframe', background='#2b2b2b')
                        style.configure('TLabelframe.Label', background='#2b2b2b', foreground='#ffffff')
                        style.configure('Status.TLabel', background='#3c3f41', foreground='#ffffff')
            except Exception:
                pass
    
    def browse_input(self):
        """Otvara dijalog za odabir ulazne datoteke ili mape."""
        if self.batch_mode.get():
            path = filedialog.askdirectory(title="Odaberi folder s SRT datotekama")
        else:
            path = filedialog.askopenfilename(
                title="Odaberite .srt datoteku",
                filetypes=[
                    ("Subtitle files", "*.srt"),
                    ("All files", "*.*")
                ]
            )
            
        if path:
            self.input_path.set(path)
            # Ako nije postavljen izlaz, postavi automatski
            if not self.output_path.get():
                if self.batch_mode.get():
                    self.output_path.set(os.path.join(os.path.dirname(path), "prevedeno"))
                else:
                    base, ext = os.path.splitext(path)
                    self.output_path.set(f"{base}_hr{ext}")
    
    def browse_output(self):
        """Otvara dijalog za odabir izlazne datoteke ili mape."""
        if self.batch_mode.get():
            path = filedialog.askdirectory(
                title="Odaberite izlaznu mapu",
                mustexist=False # Dozvoli kreiranje nove mape
            )
        else:
            default_name = ""
            if self.input_path.get():
                base, ext = os.path.splitext(self.input_path.get())
                default_name = f"{base}_hr{ext if ext else '.srt'}"
                
            path = filedialog.asksaveasfilename(
                title="Spremi prevedenu datoteku kao",
                defaultextension=".srt",
                initialfile=os.path.basename(default_name) if default_name else "",
                initialdir=os.path.dirname(default_name) if default_name else "",
                filetypes=[
                    ("Subtitle files", "*.srt"),
                    ("All files", "*.*")
                ]
            )
        
        if path:
            self.output_path.set(path)
    
    def browse_csv(self):
        """Otvara dijalog za odabir CSV datoteke s metapodacima."""
        initial_dir = ""
        if self.input_path.get():
            initial_dir = os.path.dirname(self.input_path.get())
            
        path = filedialog.askopenfilename(
            title="Odaberite CSV datoteku s metapodacima",
            initialdir=initial_dir,
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if path:
            self.metadata_csv_path.set(path)
    
    def browse_user_dict(self):
        """Otvara dijalog za odabir korisničkog rječnika."""
        initial_dir = ""
        if self.input_path.get():
            initial_dir = os.path.dirname(self.input_path.get())
            
        path = filedialog.askopenfilename(
            title="Odaberite datoteku s korisničkim rječnikom",
            initialdir=initial_dir,
            filetypes=[
                ("Text files", "*.txt;*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if path:
            self.user_dict_path.set(path)
    
    def start_translation(self):
        """Pokreće proces prevođenja."""
        input_path = self.input_path.get().strip()
        output_path = self.output_path.get().strip()
        
        # Validacija unosa
        if not input_path:
            messagebox.showerror("Greška", "Molimo odaberite ulaznu datoteku ili mapu!")
            return
            
        if not os.path.exists(input_path):
            messagebox.showerror("Greška", f"Odabrana datoteka/mapa ne postoji:\n{input_path}")
            return
            
        if not output_path:
            messagebox.showerror("Greška", "Molimo odredite izlaznu datoteku ili mapu!")
            return
            
        # Ako je batch mod, provjeri izlaznu mapu
        if self.batch_mode.get() and os.path.isfile(output_path):
            messagebox.showerror("Greška", "U batch modu izlaz mora biti mapa, ne datoteka!")
            return
            
        # Ako nije batch mod, provjeri ekstenziju
        if not self.batch_mode.get() and not output_path.lower().endswith('.srt'):
            output_path += '.srt'
            self.output_path.set(output_path)
        
        # Ažuriraj sučelje
        self.status_var.set("Priprema za prevođenje...")
        self.progress_var.set(0)
        self.progress.configure(mode='determinate')
        self.translate_btn.config(state=tk.DISABLED)
        
        # Dodaj obavijest o prevođenju
        if hasattr(self, 'progress_label'):
            self.progress_label.destroy()
            
        self.progress_label = ttk.Label(
            self.root, 
            text="Bez brige, aplikacija nije zamrznuta - u tijeku je prevođenje titlova.\nOvo može potrajati nekoliko minuta, ovisno o veličini datoteke i brzini vašeg računala.",
            foreground='red',
            wraplength=700,
            justify=tk.CENTER
        )
        self.progress_label.pack(pady=10, fill=tk.X, padx=20)
        
        self.root.update_idletasks()
        
        # Prikaži statusnu poruku s detaljima
        file_count = 0
        is_batch = self.batch_mode.get() and os.path.isdir(input_path)
        
        if is_batch:
            srt_files = [f for f in os.listdir(input_path) if f.lower().endswith('.srt')]
            file_count = len(srt_files)
        else:
            file_count = 1
            
        self.status_var.set(f"Prevođenje u tijeku (0/{file_count})")
        
        # Pokreni prevođenje u zasebnoj niti
        try:
            if is_batch:
                # Batch mod - više datoteka
                self.progress.configure(mode='determinate', maximum=100) # Maximum je 100%
                self.progress['value'] = 0
                thread = threading.Thread(target=self.run_batch_translation, daemon=True)
            else:
                # Single datoteka
                input_file = input_path
                output_file = output_path # Koristi definirani output
                self.translate_btn.config(state=tk.DISABLED)
                self.status_var.set("Prevođenje u tijeku...")
                self.progress.configure(mode='indeterminate')
                self.progress.start(10)
                thread = threading.Thread(
                    target=self.run_translation, 
                    args=(input_file, output_file), 
                    daemon=True
                )
            
            thread.start()
            self.check_thread(thread)
            
        except Exception as e:
            self.on_translation_error(str(e))
    
    def format_time(self, seconds):
        """Formatira sekunde u čitljiv vremenski format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def update_progress(self, progress, is_batch=False, current_file=None, total_files=None):
        """Ažurira statusnu traku s postotkom i procijenjenim preostalim vremenom."""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_time = current_time
            self.last_update_time = current_time
            self.last_progress = 0
            self.estimated_total_time = 0
        
        elapsed = current_time - self.start_time
        
        # Izračunaj brzinu i procijenjeno preostalo vrijeme
        if progress > self.last_progress and current_time > self.last_update_time:
            time_delta = current_time - self.last_update_time
            progress_delta = progress - self.last_progress
            
            speed = progress_delta / time_delta  # % po sekundi
            
            if speed > 0:
                remaining_progress = 100 - progress
                remaining_time = remaining_progress / speed
                self.estimated_total_time = elapsed + remaining_time
                time_str = self.format_time(remaining_time)
                
                if is_batch:
                    status_text = f"Prevođenje {current_file}/{total_files}: {progress:.1f}% (Preostalo: {time_str})"
                else:
                    status_text = f"Prevođenje: {progress:.1f}% (Preostalo: {time_str})"
                
                self.root.after(0, self.status_var.set, status_text)
        
        elif progress == 0 and elapsed > 1:
            self.root.after(0, self.status_var.set, f"Prevođenje... (Proteklo: {self.format_time(elapsed)})")
            
        self.root.after(0, self.progress_var.set, progress)
        
        self.last_update_time = current_time
        self.last_progress = progress

    def run_translation(self, input_file, output_file):
        """Pokreće prevođenje jedne datoteke."""
        try:
            # Provjeri postoji li već prevedena datoteka
            if os.path.exists(output_file):
                base, ext = os.path.splitext(output_file)
                if not ext.lower() == '.srt':
                    output_file = f"{base}.srt"
                
                # Ako datoteka već postoji, prikaži upozorenje
                result = messagebox.askyesno(
                    "Datoteka već postoji",
                    f"Izlazna datoteka već postoji:\n{output_file}\n\nŽelite li je zamijeniti?",
                    parent=self.root
                )
                
                if not result:
                    self.root.after(0, self.reset_ui_after_translation)
                    return
            
            self.start_time = None
            self.last_progress = 0
            self.progress.configure(mode='determinate')
            
            metadata_csv = self.metadata_csv_path.get().strip()
            user_dict_path = self.user_dict_path.get().strip()
            
            translator = ImprovedSubtitleTranslator(
                metadata_csv=metadata_csv if metadata_csv else None,
                user_dict_path=user_dict_path if user_dict_path else None
            )
            translator._gem_enabled = self.gemini_enabled.get()
            
            # Ovdje bi trebala biti logika za update progress-a ako translator to podržava
            # Za sada, simuliramo neodređeni progress za single file
            self.progress.configure(mode='indeterminate')
            self.progress.start(10)
            
            translator.translate_file(input_file, output_file)
            
            self.progress.stop()
            self.progress.configure(mode='determinate')
            self.progress_var.set(100)
            
            self.root.after(0, self.on_translation_success)
            
        except Exception as e:
            self.root.after(0, self.on_translation_error, str(e))
            
    def run_batch_translation(self):
        """Pokreće batch prevođenje (više datoteka)."""
        try:
            self.start_time = None
            self.last_progress = 0
            
            input_dir = self.input_path.get()
            output_dir = self.output_path.get()
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            srt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.srt')]
            total_files = len(srt_files)
            if total_files == 0:
                self.root.after(0, self.on_translation_error, "Nema .srt datoteka u mapi.")
                return

            self.progress.configure(mode='determinate', maximum=100)
            
            metadata_csv = self.metadata_csv_path.get().strip()
            user_dict_path = self.user_dict_path.get().strip()
            
            translator = ImprovedSubtitleTranslator(
                metadata_csv=metadata_csv if metadata_csv else None,
                user_dict_path=user_dict_path if user_dict_path else None
            )
            translator._gem_enabled = self.gemini_enabled.get()
            
            success_count = 0
            for i, srt_file in enumerate(srt_files):
                input_file = os.path.join(input_dir, srt_file)
                output_file = os.path.join(output_dir, f"{Path(srt_file).stem}_hr.srt")
                
                # Izračunaj napredak
                progress = ((i + 1) / total_files) * 100
                
                # Ažuriraj GUI
                self.root.after(0, lambda p=progress, i=i, t=total_files: self.update_progress(
                    p, is_batch=True, current_file=i+1, total_files=t
                ))
                
                try:
                    translated = translator.translate_file(input_file, output_file)
                    if translated:
                        success_count += 1
                except Exception as e:
                    print(f"Greška pri prevođenju {srt_file}: {e}")
            
            self.root.after(0, lambda: self.on_translation_complete(success_count, total_files))
            
        except Exception as e:
            self.root.after(0, self.on_translation_error, str(e))

    def toggle_advanced(self, show=None):
        """Prikazuje ili skriva dodatne opcije."""
        self.show_advanced = not self.show_advanced if show is None else show
        
        if self.show_advanced:
            self.advanced_frame.grid(row=4, column=0, columnspan=3, sticky='ew', pady=5, padx=5)
            self.advanced_btn.config(text="▲ Sakrij opcije")
            # Poveži tooltips za napredne opcije tek sada
            adv_entries = self.advanced_frame.winfo_children()[0].winfo_children()
            try:
                ToolTip(adv_entries[1], self.tooltips['metadata_csv']) # Entry za CSV
                ToolTip(adv_entries[4], self.tooltips['user_dict'])  # Entry za User Dict
            except IndexError:
                pass # Zaštita ako se raspored promijeni
        else:
            self.advanced_frame.grid_remove()
            self.advanced_btn.config(text="▼ Dodatne opcije")
    
    def toggle_batch_mode(self):
        """Ažurira GUI labele za batch ili single mod."""
        is_batch = self.batch_mode.get()
        btn_text_input = "Odaberi mapu..." if is_batch else "Odaberi datoteku..."
        btn_text_output = "Odaberi mapu..." if is_batch else "Odaberi datoteku..."
        
        self.input_btn.config(text=btn_text_input)
        self.output_btn.config(text=btn_text_output)
        
        # Automatski predloži izlaz
        self.auto_suggest_output()

    def auto_suggest_output(self):
        """Automatski predlaže izlaznu putanju na temelju ulazne."""
        input_path = self.input_path.get()
        if not input_path:
            return
            
        if self.batch_mode.get():
            if os.path.isdir(input_path):
                self.output_path.set(os.path.join(input_path, "prevedeno"))
            elif os.path.isfile(input_path):
                self.output_path.set(os.path.join(os.path.dirname(input_path), "prevedeno"))
        else:
            if os.path.isfile(input_path):
                base, ext = os.path.splitext(input_path)
                self.output_path.set(f"{base}.hr.srt")
            elif os.path.isdir(input_path):
                # Ako je odabrana mapa ali nije batch, ponudi spremanje kao...
                self.output_path.set(os.path.join(input_path, "prevedeno.hr.srt"))
    
    def check_thread(self, thread):
        """Provjerava je li dretva (thread) za prevođenje još aktivna."""
        if thread.is_alive():
            self.root.after(100, self.check_thread, thread)
        else:
            # Dretva je gotova, vrati gumb u normalno stanje
            self.translate_btn.config(state=tk.NORMAL)
            # Zaustavi indeterminate progress bar ako je bio aktivan
            if self.progress.cget('mode') == 'indeterminate':
                self.progress.stop()
                self.progress_var.set(100) # Postavi na 100 da izgleda dovršeno
                
    def on_translation_success(self):
        """Poziva se kada je prevođenje jedne datoteke uspješno."""
        self.status_var.set("Prijevod uspješno završen!")
        self.progress_var.set(100)
        messagebox.showinfo("Gotovo", "Prijevod je uspješno završen!")
        self.reset_ui_after_translation()

    def on_translation_complete(self, success_count, total_files):
        """Poziva se nakon završetka batch prevođenja."""
        self.status_var.set(f"Batch gotov: {success_count}/{total_files} prevedeno.")
        self.progress_var.set(100)
        messagebox.showinfo("Gotovo", f"Batch prevođenje završeno.\nPrevedeno: {success_count}/{total_files}")
        self.reset_ui_after_translation()

    def on_translation_error(self, error_msg):
        """Poziva se u slučaju greške."""
        self.status_var.set(f"Greška: {error_msg}")
        self.progress.stop()
        self.progress_var.set(0)
        messagebox.showerror("Greška", f"Došlo je do greške tijekom prevođenja:\n\n{error_msg}")
        self.translate_btn.config(state=tk.NORMAL)

    def reset_ui_after_translation(self):
        """Resetira UI nakon 2 sekunde."""
        def reset():
            self.status_var.set("Spreman za prevođenje")
            self.progress_var.set(0)
            self.translate_btn.config(state=tk.NORMAL)
        self.root.after(2000, reset)

if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleTranslatorApp(root)
    # Poveži auto-suggest na promjenu ulaznog polja
    app.input_path.trace_add("write", lambda *args: app.auto_suggest_output())
    root.mainloop()

