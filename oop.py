import re
import os
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer

def parse_srt(file_path):
    """Parse SRT file and return list of subtitle entries."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines to separate subtitle blocks
    blocks = re.split(r'\n\n+', content.strip())
    
    subtitles = []
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            index = lines[0].strip()
            timestamp = lines[1].strip()
            text = '\n'.join(lines[2:])
            subtitles.append({
                'index': index,
                'timestamp': timestamp,
                'text': text
            })
    
    return subtitles

def detect_gender_hints(text):
    """Detect gender hints from the text."""
    # Često korištene zamjenice i riječi koje indiciraju rod
    female_indicators = ['she', 'her', 'hers', 'herself', 'lady', 'queen', 'princess', 'mother', 'daughter', 'sister']
    male_indicators = ['he', 'his', 'him', 'himself', 'lord', 'king', 'prince', 'father', 'son', 'brother']
    
    text_lower = text.lower()
    
    # Provjeri za ženske indikatore
    if any(indicator in text_lower.split() for indicator in female_indicators):
        return "<female>"
    # Provjeri za muške indikatore
    elif any(indicator in text_lower.split() for indicator in male_indicators):
        return "<male>"
    return ""

def translate_batch(texts, model, tokenizer, previous_texts=None, max_length=512):
    """Translate a batch of texts together for better context."""
    try:
        # Pripremi tekstove s kontekstom i oznakama roda
        texts_with_context = []
        for i, text in enumerate(texts):
            # Pripremi kontekst ako postoji
            context = ""
            if previous_texts and i < len(previous_texts):
                # Uzmi samo relevantne dijelove prethodnog teksta za kontekst
                prev_text = previous_texts[i]
                # Uzmi samo zadnju rečenicu iz prethodnog teksta ako postoji više rečenica
                prev_sentences = re.split(r'[.!?]+\s*', prev_text)
                if prev_sentences:
                    context = prev_sentences[-1]
            
            # Detektiraj oznake roda
            gender_hint = detect_gender_hints(text)
            if not gender_hint and context:
                gender_hint = detect_gender_hints(context)
            
            # Složi tekst s kontekstom i oznakom roda, koristeći suptilniji separator
            full_text = f">>hrv<< {gender_hint} {context + '; ' if context else ''}{text}".strip()
            texts_with_context.append(full_text)
        
        # Tokeniziraj sve tekstove odjednom
        inputs = tokenizer(texts_with_context, 
                         return_tensors="pt", 
                         padding=True, 
                         truncation=True, 
                         max_length=max_length)
        
        # Generiraj prijevode za cijeli batch
        translated = model.generate(**inputs, max_length=max_length)
        
        # Dekodiraj sve prijevode
        translated_texts = [tokenizer.decode(t, skip_special_tokens=True) 
                          for t in translated]
        
        # Očisti prijevode od kontekstualnih oznaka
        cleaned_texts = []
        for text in translated_texts:
            # Ukloni oznake za kontekst
            text = re.sub(r'\s*>>>\s*', ' ', text)
            # Ukloni oznake za rod
            text = re.sub(r'\s*<(male|female)>\s*', ' ', text)
            # Ukloni višestruke razmake
            text = re.sub(r'\s+', ' ', text)
            # Ukloni razmake na početku i kraju
            text = text.strip()
            cleaned_texts.append(text)
        
        return cleaned_texts
    except Exception as e:
        print(f"     ⚠️  Greška u batch prevođenju: {e}")
        return translate_texts_individually(texts, model, tokenizer)

def translate_texts_individually(texts, model, tokenizer, max_length=512):
    """Fallback: translate each text individually."""
    results = []
    for text in texts:
        text_with_token = ">>hrv<< " + text
        inputs = tokenizer(text_with_token, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        translated = model.generate(**inputs, max_length=max_length)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        results.append(translated_text)
    return results

def write_srt(subtitles, output_path):
    """Write subtitles to SRT file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles):
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['timestamp']}\n")
            f.write(f"{sub['text']}\n")
            if i < len(subtitles) - 1:
                f.write("\n")

def find_srt_files(folder_path):
    """Find all SRT files in the given folder."""
    folder = Path(folder_path)
    return list(folder.glob("*.srt"))

def translate_srt_file(input_path, model, tokenizer, batch_size=5):
    """Translate a single SRT file with context-aware batching."""
    input_path = Path(input_path)
    
    # Kreiraj ime output filea
    output_name = input_path.stem + ".hr" + input_path.suffix
    output_path = input_path.parent / output_name
    
    # Provjeri da li već postoji prevedena verzija
    if output_path.exists():
        print(f"  ⚠️  Preskačem - već postoji: {output_name}")
        return False
    
    print(f"  📄 Prevođenje: {input_path.name}")
    
    # Parse titlove
    subtitles = parse_srt(input_path)
    print(f"     Pronađeno {len(subtitles)} titlova")
    
    # Prevedi u grupama (batch)
    i = 0
    while i < len(subtitles):
        # Uzmi sljedećih batch_size titlova
        batch_end = min(i + batch_size, len(subtitles))
        batch_subtitles = subtitles[i:batch_end]
        
        # Izvuci tekstove i prethodne tekstove za kontekst
        batch_texts = [sub['text'] for sub in batch_subtitles]
        
        # Uzmi prethodne titlove za kontekst
        previous_texts = []
        for idx in range(i, batch_end):
            if idx > 0:
                previous_texts.append(subtitles[idx-1]['text'])
            else:
                previous_texts.append("")
        
        # Prevedi batch s kontekstom
        translated_texts = translate_batch(batch_texts, model, tokenizer, previous_texts)
        
        # Dodijeli prevedene tekstove natrag
        for j, translated_text in enumerate(translated_texts):
            subtitles[i + j]['text'] = translated_text
        
        i = batch_end
        
        if i % 25 == 0:
            print(f"     Prevedeno {i}/{len(subtitles)}...")
    
    # Spremi prevedene titlove
    write_srt(subtitles, output_path)
    print(f"  ✓ Spremljeno kao: {output_name}\n")
    return True

def main():
    # Upiši putanju do foldera s titlovima
    folder_path = input("Unesi putanju do foldera s SRT titlovima: ").strip().strip('"')
    
    # Provjeri da li folder postoji
    if not os.path.exists(folder_path):
        print(f"❌ Greška: Folder '{folder_path}' ne postoji!")
        return
    
    # Pronađi sve SRT fileove
    srt_files = find_srt_files(folder_path)
    
    if not srt_files:
        print(f"❌ Nema SRT fileova u folderu '{folder_path}'")
        return
    
    # Isključi već prevedene fileove i hrvatske titlove
    # Preskače: .hr.srt, .hrv.srt, .cro.srt ili bilo gdje u imenu ima hrv/cro
    srt_files = [f for f in srt_files 
                 if not f.stem.endswith('.hr') 
                 and 'hrv' not in f.stem.lower() 
                 and 'cro' not in f.stem.lower()]
    
    if not srt_files:
        print("❌ Nema novih fileova za prevođenje (svi su već prevedeni)")
        return
    
    print(f"\n📁 Pronađeno {len(srt_files)} SRT file(ova) za prevođenje\n")
    
    print("🔄 Učitavanje modela (ovo može potrajati pri prvom pokretanju)...")
    # Koristi javno dostupni model za prijevod
    model_name = "Helsinki-NLP/opus-mt-en-zls"
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print("✓ Model učitan!\n")
    
    # Prevedi sve fileove
    translated_count = 0
    batch_size = 3  # Broj titlova koji se prevode zajedno za bolji kontekst
    
    for i, srt_file in enumerate(srt_files, 1):
        print(f"[{i}/{len(srt_files)}]")
        if translate_srt_file(srt_file, model, tokenizer, batch_size):
            translated_count += 1
    
    print(f"\n{'='*50}")
    print(f"✓ GOTOVO! Prevedeno {translated_count}/{len(srt_files)} fileova")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()