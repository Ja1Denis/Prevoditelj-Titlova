"""
SRT Filler Words Analyzer - Gemini 2.5 Flash
Analizira ispunjavače pauze u hrvatskim titlovima
"""

import os
import re
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# =============================================================================
# UČITAJ API KLJUČ IZ .ENV FILEA
# =============================================================================

# Učitaj environment varijable iz .env filea
load_dotenv()

# Dohvati API ključ
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY nije pronađen u .env fileu!")

# Inicijaliziraj Gemini klijent
genai.configure(api_key=API_KEY)

print(f"✅ API ključ učitan iz .env filea")

# =============================================================================
# LISTA ISPUNJAVAČA PAUZE
# =============================================================================

FILLERS_DATABASE = {
    "Osnovni": ["pa", "eto", "znači", "ma", "e", "e pa", "dobro", "dakle", "daklem", "onda"],
    "Zvučni": ["mmm", "hmm", "eee", "aaaa", "um", "uhm", "eh"],
    "Diskursni": ["gledaj", "vidi", "slušaj", "gle", "evo", "hajde", "čuj", "stani", "stoj"],
    "Kolokvijalizam": ["kao", "tipa", "ono", "jel", "šta", "nego", "ajde", "de", "kužiš", "razumiješ"],
    "Uvodni": ["naime", "međutim", "inače", "uostalom", "uglavnom", "ukratko", "recimo"],
    "Vremenski": ["zatim", "potom", "sad", "sada", "trenutno", "odmah", "prvo", "drugo"],
    "Korekcijski": ["to jest", "odnosno", "točnije", "bolje rečeno", "zapravo", "u biti"],
    "Emocionalni": ["bogami", "bome", "naravno", "svakako", "sigurno", "valjda", "možda"]
}

# =============================================================================
# FUNKCIJE ZA SRT OBRADU
# =============================================================================

def parse_srt(srt_content):
    """Parsira SRT datoteku i vraća listu subtitlova"""
    pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)'
    matches = re.findall(pattern, srt_content, re.DOTALL)
    
    subtitles = []
    for match in matches:
        subtitles.append({
            'index': int(match[0]),
            'start': match[1],
            'end': match[2],
            'text': match[3].strip()
        })
    return subtitles

def read_srt_file(filepath):
    """Čita SRT datoteku"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# =============================================================================
# GEMINI ANALIZA
# =============================================================================

def analyze_fillers_with_gemini(text, context=""):
    """Koristi gemini-1.5-flash za detekciju ispunjavača"""
    
    # Kreiraj formatiranu listu ispunjavača za prompt
    fillers_list = "\n".join([
        f"{category}: {', '.join(words)}"
        for category, words in FILLERS_DATABASE.items()
    ])
    
    # System prompt
    prompt = f"""
Ti si lingvistički ekspert za hrvatski jezik specijaliziran za diskursne markere i ispunjavače pauze.

ZADATAK: Analiziraj sljedeći tekst i pronađi SVE ispunjavače pauze.

KATEGORIJE ISPUNJAVAČA:
{fillers_list}

TEKST ZA ANALIZU:
"{text}"

KONTEKST SCENE (ako postoji):
{context if context else "Nema dodatnog konteksta"}

VRATI REZULTAT U JSON FORMATU:
{{
    "fillers_found": [
        {{
            "word": "pronađeni ispunjavač",
            "category": "kategorija",
            "position": pozicija_u_tekstu,
            "context": "zašto je to ispunjavač u ovom kontekstu"
        }}
    ],
    "total_count": broj_ukupno,
    "analysis": "kratak komentar o upotrebi ispunjavača u ovom tekstu"
}}

VAŽNO: 
- Prepoznaj ispunjavače I KADA SU U RAZLIČITOM OBLIKU (npr. "gledaš" od "gledaj")
- Obrati pažnju na kontekst - "dobro" može biti ispunjavač ili prilog
- Vrati SAMO JSON, bez dodatnog teksta
"""

    try:
        # Poziv Gemini API-ja
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2048,
            )
        )
        
        # Izvuci JSON iz odgovora
        result_text = response.text.strip()
        
        # Ukloni markdown code blocks ako postoje
        if result_text.startswith("```
            result_text = result_text.split("```json").split("```
        elif result_text.startswith("```"):
            result_text = result_text.split("``````")[0].strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        print(f"⚠️  Greška u Gemini analizi: {e}")
        return {
            "fillers_found": [],
            "total_count": 0,
            "analysis": f"Greška: {str(e)}"
        }

# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def analyze_srt_file(filepath, output_file="filler_analysis.json", max_subtitles=10):
    """Analizira SRT datoteku"""
    
    print(f"📂 Čitam datoteku: {filepath}")
    srt_content = read_srt_file(filepath)
    
    print("🔍 Parsiram titlove...")
    subtitles = parse_srt(srt_content)
    
    print(f"✅ Pronađeno {len(subtitles)} titlova")
    print(f"🤖 Analiziram prvih {max_subtitles} s Gemini 2.5 Flash...\n")
    
    results = []
    total_fillers = 0
    
    for i, subtitle in enumerate(subtitles[:max_subtitles], 1):
        print(f"[{i}/{max_subtitles}] Analiziram: {subtitle['text'][:50]}...")
        
        # Uzmi kontekst (prethodni i sljedeći titl)
        context = ""
        if i > 1:
            context += f"Prethodno: {subtitles[i-2]['text']}\n"
        if i < len(subtitles):
            context += f"Sljedeće: {subtitles[i]['text']}"
        
        analysis = analyze_fillers_with_gemini(subtitle['text'], context)
        
        if analysis['total_count'] > 0:
            results.append({
                'subtitle_index': subtitle['index'],
                'timecode': f"{subtitle['start']} --> {subtitle['end']}",
                'text': subtitle['text'],
                'analysis': analysis
            })
            total_fillers += analysis['total_count']
            print(f"   ✓ Pronađeno: {analysis['total_count']} ispunjavača")
        else:
            print(f"   - Nema ispunjavača")
    
    # Spremi rezultate
    output = {
        'summary': {
            'total_subtitles_analyzed': max_subtitles,
            'subtitles_with_fillers': len(results),
            'total_fillers_found': total_fillers
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Analiza gotova!")
    print(f"📊 Ukupno ispunjavača: {total_fillers}")
    print(f"💾 Rezultati spremljeni u: {output_file}")
    
    return output

# =============================================================================
# POKRETANJE
# =============================================================================

if __name__ == "__main__":
    # Primjer uporabe
    SRT_FILE = "Emmanuelle.2024.1080p.WEBRip.x264.AAC5.1-LAMA.hr.srt"
    
    try:
        results = analyze_srt_file(SRT_FILE, max_subtitles=20)
        
        # Ispiši primjer rezultata
        print("\n" + "="*60)
        print("PRIMJER REZULTATA:")
        print("="*60)
        if results['results']:
            example = results['results'][0]
            print(f"\nTitl #{example['subtitle_index']}")
            print(f"Vrijeme: {example['timecode']}")
            print(f"Tekst: {example['text']}")
            print(f"\nPronađeni ispunjavači:")
            for filler in example['analysis']['fillers_found']:
                print(f"  • '{filler['word']}' ({filler['category']})")
                print(f"    Razlog: {filler['context']}")
    
    except FileNotFoundError:
        print(f"❌ Datoteka {SRT_FILE} nije pronađena!")
    except Exception as e:
        print(f"❌ Greška: {e}")
