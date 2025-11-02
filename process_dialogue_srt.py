from improved_translator import ImprovedSubtitleTranslator
import re

def parse_srt_content(content):
    """Parsira SRT sadržaj i vraća listu rječnika s podacima o titlovima."""
    blocks = content.strip().split('\n\n')
    subtitles = []
    
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 2:  # Trebamo barem broj i vremenski kod
            # Ako prvi red sadrži samo broj, ukloni ga
            if lines[0].isdigit():
                lines = lines[1:]
                
            if len(lines) >= 2:  # Nakon uklanjanja broja, provjeri opet
                timestamp = lines[0]
                text = '\n'.join(lines[1:])
                subtitles.append({
                    'timestamp': timestamp,
                    'text': text,
                    'original_text': text
                })
    
    return subtitles

def process_dialogue():
    # Inicijaliziraj prevoditelj
    translator = ImprovedSubtitleTranslator()
    
    # SRT sadržaj
    srt_content = """7
00:06:10,287 --> 00:06:12,081
[IN ENGLISH]:
Do you need anything?
 
8
00:06:12,581 --> 00:06:13,582
No, I'm fine.
 
9
00:06:43,570 --> 00:06:44,696
[MAN]:
Breakfast, ma'am.
 
10
00:06:44,822 --> 00:06:46,824
[EMMANUELLE]:
You can leave it in the living room.
 
11
00:07:01,713 --> 00:07:03,799
How long have you been working here?
 
12
00:07:05,300 --> 00:07:06,260
Um...
 
13
00:07:07,886 --> 00:07:09,096
Six months.
 
14
00:07:10,681 --> 00:07:11,849
And you like it?
 
15
00:07:14,184 --> 00:07:15,227
Yes.
 
16
00:07:15,477 --> 00:07:16,645
Why?
 
17
00:07:18,564 --> 00:07:19,690
Why what?
 
18
00:07:20,232 --> 00:07:21,608
Why do you like it?
 
19
00:07:22,317 --> 00:07:23,277
Well...
 
20
00:07:23,819 --> 00:07:25,571
we work long hours
 
21
00:07:26,029 --> 00:07:28,782
but the pay is pretty good,
and the hotel is beautiful.
 
22
00:07:30,659 --> 00:07:32,077
What about the clientele?
 
23
00:07:33,829 --> 00:07:35,038
What about them?
 
24
00:07:35,664 --> 00:07:37,040
Are they to your taste?
 
25
00:07:40,335 --> 00:07:41,295
Well...
 
26
00:07:42,087 --> 00:07:44,798
I like to put myself
at the guests' disposal.
 
27
00:07:48,177 --> 00:07:49,052
So?
"""

    # Parsiraj SRT sadržaj
    subtitles = parse_srt_content(srt_content)
    
    # Obradi svaki titl s kontekstom
    total = len(subtitles)
    for i in range(total):
        # Dohvati kontekst za trenutni titl
        prev_texts = [subtitles[j]['text'] for j in range(max(0, i-2), i)]
        next_texts = [subtitles[j]['text'] for j in range(i+1, min(i+3, total))]
        
        # Prikaži napredak
        print(f"\rObrada titla {i+1}/{total}...", end='', flush=True)
        
        # Obradi svaki red u titlu posebno
        lines = subtitles[i]['text'].split('\n')
        processed_lines = []
        
        for line in lines:
            if line.strip():  # Ako linija nije prazna
                # Obradi liniju s kontekstom
                processed_line = translator._process_discourse_marker(
                    text=line,
                    context={
                        'prev_texts': prev_texts,
                        'next_texts': next_texts,
                        'is_first_line': len(processed_lines) == 0,
                        'is_last_line': line == lines[-1]
                    }
                )
                processed_lines.append(processed_line)
            else:
                processed_lines.append('')
        
        # Spoji linije natrag u jedan tekst
        subtitles[i]['text'] = '\n'.join(processed_lines)
        
        # Ažuriraj kontekst za sljedeće titlove
        if i + 1 < total:
            next_texts[0] = subtitles[i+1]['text']
    
    # Spremi u novu SRT datoteku
    output_file = 'processed_dialogue.srt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{sub['timestamp']}\n")
            f.write(f"{sub['text']}\n\n")
    
    print(f"\n\n{'='*50}")
    print(f"✓ Obrada završena! Rezultat spremljen u '{output_file}'.")
    print(f"{'='*50}\n")
    
    # Prikaži sažetak promjena
    changed = 0
    for i, sub in enumerate(subtitles):
        if sub['text'] != sub.get('original_text', ''):
            changed += 1
            if changed == 1:
                print("\nDetaljni pregled promjena:")
                print("-" * 50)
            
            print(f"Titl {i+1}:")
            print(f"  Original: {sub['original_text']}")
            print(f"  Prevedeno: {sub['text']}")
            print("-" * 50)
    
    print(f"\nUkupno promijenjeno {changed} od {len(subtitles)} titlova.")
    print("\nNapomene:")
    print("- Disklejmeri poput 'Well...' sada se konzistentno prevode u 'Pa...'")
    print("- Očuvava se formatiranje i vremenski kodovi")
    print("- Kontekstualne promjene su označene u izvještaju")

if __name__ == '__main__':
    process_dialogue()
