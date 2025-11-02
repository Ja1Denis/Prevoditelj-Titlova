from improved_translator import ImprovedSubtitleTranslator
import re

def parse_srt_content(content):
    """Parsira SRT sadržaj i vraća listu rječnika s podacima o titlovima."""
    blocks = content.strip().split('\n\n')
    subtitles = []
    
    for block in blocks:
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) >= 2:  # Trebamo barem broj i vremenski kod
            # Prvi red je broj, drugi je vremenski kod, ostalo je tekst
            index = lines[0]
            timestamp = lines[1]
            text = '\n'.join(lines[2:]) if len(lines) > 2 else ""
            
            subtitles.append({
                'index': index,
                'timestamp': timestamp,
                'text': text,
                'original_text': text
            })
    
    return subtitles

def process_dialogue():
    # Inicijaliziraj prevoditelj
    translator = ImprovedSubtitleTranslator()
    
    # SRT sadržaj
    srt_content = """1
00:06:10,287 --> 00:06:12,081
[IN ENGLISH]:
Do you need anything?

2
00:06:12,581 --> 00:06:13,582
No, I'm fine.

3
00:06:43,570 --> 00:06:44,696
[MAN]:
Breakfast, ma'am.

4
00:06:44,822 --> 00:06:46,824
[EMMANUELLE]:
You can leave it in the living room.

5
00:07:01,713 --> 00:07:03,799
How long have you been working here?

6
00:07:05,300 --> 00:07:06,260
Um...

7
00:07:07,886 --> 00:07:09,096
Six months.

8
00:07:10,681 --> 00:07:11,849
And you like it?

9
00:07:14,184 --> 00:07:15,227
Yes.

10
00:07:15,477 --> 00:07:16,645
Why?

11
00:07:18,564 --> 00:07:19,690
Why what?

12
00:07:20,232 --> 00:07:21,608
Why do you like it?

13
00:07:22,317 --> 00:07:23,277
Well...

14
00:07:23,819 --> 00:07:25,571
we work long hours

15
00:07:26,029 --> 00:07:28,782
but the pay is pretty good,
and the hotel is beautiful.

16
00:07:30,659 --> 00:07:32,077
What about the clientele?

17
00:07:33,829 --> 00:07:35,038
What about them?

18
00:07:35,664 --> 00:07:37,040
Are they to your taste?

19
00:07:40,335 --> 00:07:41,295
Well...

20
00:07:42,087 --> 00:07:44,798
I like to put myself
at the guests' disposal.

21
00:07:48,177 --> 00:07:49,052
So?
"""

    # Parsiraj SRT sadržaj
    subtitles = parse_srt_content(srt_content)
    
    print(f"Pronađeno {len(subtitles)} titlova za obradu...")
    
    # Obradi svaki titl
    for i, sub in enumerate(subtitles):
        # Dohvati kontekst za trenutni titl
        context = {
            'prev_texts': [subtitles[j]['text'] for j in range(max(0, i-2), i) if j < len(subtitles)],
            'next_texts': [subtitles[j]['text'] for j in range(i+1, min(i+3, len(subtitles)))]
        }
        
        print(f"\n--- Obrada titla {i+1} ---")
        print(f"Original: {sub['text']}")
        
        # Obradi disklejmere i ispunjivače
        processed_text = translator._process_discourse_marker(
            text=sub['text'],
            context=context
        )
        
        print(f"Obradjeno: {processed_text}")
        
        # Ažuriraj tekst u titlu
        sub['text'] = processed_text
    
    # Spremi u novu SRT datoteku
    output_file = 'processed_dialogue_fixed.srt'
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles, 1):
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['timestamp']}\n")
            f.write(f"{sub['text']}\n")
            if i < len(subtitles):
                f.write("\n")  # Dodaj prazan red između titlova, ali ne na kraju
    
    print(f"\n✓ Obrada završena! Rezultat spremljen u '{output_file}'.")
    
    # Ispiši promjene
    print("\nPregled promjena:")
    print("-" * 50)
    for i, sub in enumerate(subtitles):
        if sub['text'] != sub.get('original_text', ''):
            print(f"Titl {i+1}:")
            print(f"  Original: {sub['original_text']}")
            print(f"  Prevedeno: {sub['text']}")
            print("-" * 50)

if __name__ == '__main__':
    process_dialogue()
