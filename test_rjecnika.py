import re

def ucitaj_rjecnik(putanja):
    """Učitava rječnik iz datoteke i vraća rječnik s prijevodima"""
    rjecnik = {}
    try:
        with open(putanja, 'r', encoding='utf-8') as f:
            for linija in f:
                linija = linija.strip()
                # Preskačemo prazne linije i komentare
                if not linija or linija.startswith('#'):
                    continue
                
                # Razdvajamo na original i prijevod
                if '->' in linija:
                    original, prijevod = linija.split('->', 1)
                    original = original.strip()
                    prijevod = prijevod.strip()
                    
                    # Ako postoji kontekst u uglatim zagradama
                    if '[' in original and ']' in original:
                        kljuc = original.split('[', 1)[0].strip()
                        kontekst = original[original.find('[')+1:original.find(']')]
                        kljuc_sa_kontekstom = f"{kljuc} [{kontekst}]"
                        rjecnik[kljuc_sa_kontekstom.lower()] = prijevod
                    else:
                        rjecnik[original.lower()] = prijevod
        
        print(f"Učitano {len(rjecnik)} unosa iz rječnika.")
        return rjecnik
    except Exception as e:
        print(f"Greška pri učitavanju rječnika: {e}")
        return {}

def prevedi_tekst(tekst, rjecnik):
    """Primjenjuje prijevode iz rječnika na zadani tekst"""
    if not tekst:
        return ""
    
    # Prvo pokušavamo pronaći potpune podudaranja s kontekstom
    for kljuc, prijevod in rjecnik.items():
        if '[' in kljuc and ']' in kljuc:
            osnovni_kljuc = kljuc.split('[', 1)[0].strip()
            if osnovni_kljuc.lower() == tekst.lower():
                return prijevod
    
    # Ako nema kontekstualnog podudaranja, pokušavamo s običnim podudaranjem
    return rjecnik.get(tekst.lower(), tekst)

# Testni primjeri
testni_tekstovi = [
    "well",
    "so",
    "there you go",
    "meaning",
    "oh come on",
    "hello"  # Riječ koja nije u rječniku
]

# Učitaj rječnik
rjecnik = ucitaj_rjecnik("osnovni_rijecnik.txt")

# Testiraj prijevode
print("\nTestiranje prijevoda:" + "="*50)
for tekst in testni_tekstovi:
    prijevod = prevedi_tekst(tekst, rjecnik)
    print(f"'{tekst}' -> '{prijevod}'")

# Dodatni test s rečenicom
recenica = "Well, so there you go, that's the meaning of life."
print("\nTestiranje rečenice:" + "="*50)
print(f"Original: {recenica}")

# Podijeli rečenicu na riječi i prevedi svaku
prevedene_rijeci = []
for rijec in re.findall(r"\b[\w']+\b|\S", recenica):
    prevedene_rijeci.append(prevedi_tekst(rijec, rjecnik) or rijec)

print("Prevedeno: " + " ".join(prevedene_rijeci))
