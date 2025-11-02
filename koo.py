import re
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

def translate_text(text, model, tokenizer, max_length=512):
    """Translate text using the model."""
    # Dodaj hrvatski jezik token na početak teksta
    text_with_token = ">>hrv<< " + text
    
    # Split long text into sentences to avoid token limit
    sentences = re.split(r'([.!?]\s+)', text)
    
    translated_parts = []
    current_batch = ">>hrv<< "
    
    for sentence in sentences:
        if len(tokenizer.encode(current_batch + sentence)) < max_length:
            current_batch += sentence
        else:
            if current_batch and len(current_batch) > 8:  # Više od samog tokena
                inputs = tokenizer(current_batch, return_tensors="pt", padding=True)
                translated = model.generate(**inputs)
                translated_parts.append(tokenizer.decode(translated[0], skip_special_tokens=True))
            current_batch = ">>hrv<< " + sentence
    
    # Translate remaining text
    if current_batch and len(current_batch) > 8:
        inputs = tokenizer(current_batch, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_parts.append(tokenizer.decode(translated[0], skip_special_tokens=True))
    
    return ' '.join(translated_parts)

def write_srt(subtitles, output_path):
    """Write subtitles to SRT file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles):
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['timestamp']}\n")
            f.write(f"{sub['text']}\n")
            if i < len(subtitles) - 1:
                f.write("\n")

def main():
    # Configuration
    input_file = "titlovi/Game.of.Thrones.S04E01.1080p.BluRay.x265.10bit.6CH.ReEnc-LUMI.srt"  # Promijeni na svoj file
    output_file = "output_subtitles_hr.srt"
    
    print("Učitavanje modela...")
    # Za hrvatski se koristi 'sh' (Serbo-Croatian) kod
    # Opcije: "Helsinki-NLP/opus-mt-tc-base-en-sh" ili "Helsinki-NLP/opus-mt-tc-big-en-sh"
    # tc-big je kvalitetniji ali sporiji
    model_name = "Helsinki-NLP/opus-mt-tc-base-en-sh"
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    print(f"Učitavanje titlova iz {input_file}...")
    subtitles = parse_srt(input_file)
    print(f"Pronađeno {len(subtitles)} titlova.")
    
    print("Prevođenje u tijeku...")
    for i, sub in enumerate(subtitles):
        translated_text = translate_text(sub['text'], model, tokenizer)
        sub['text'] = translated_text
        
        if (i + 1) % 10 == 0:
            print(f"Prevedeno {i + 1}/{len(subtitles)} titlova...")
    
    print(f"Spremanje prevedenih titlova u {output_file}...")
    write_srt(subtitles, output_file)
    
    print("Gotovo! ✓")

if __name__ == "__main__":
    main()