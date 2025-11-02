# Sinkronizator titlova (v0.12)

Ova aplikacija omogućuje prevođenje i sinkronizaciju titlova s engleskog na hrvatski jezik, s naprednim značajkama poput detekcije ispuna, kontekstualnog prevođenja i prilagođenog rječnika.

## Glavne značajke

- **Prepoznavanje i obrada postojećih prijevoda** - Aplikacija prepoznaje već prevedene datoteke i nudi opciju njihove zamjene
- **Grafičko korisničko sučelje** - Intuitivno sučelje za jednostavnu uporabu
- **Batch obrada** - Mogućnost obrade više datoteka odjednom
- **Integracija s Google Gemini** - Poboljšanje kvalitete prijevoda korištenjem Google Gemini API-ja
- **Prilagodljivi rječnik** - Mogućnost dodavanja vlastitih prijevoda kroz korisnički rječnik
- **Detekcija ispuna** - Automatsko prepoznavanje i obrada ispuna u dijalozima
- **Podrška za metapodatke** - Učitavanje dodatnih informacija iz CSV datoteke

## Što aplikacija može

- **Automatsko prevođenje** - Brzo i precizno prevođenje titlova s engleskog na hrvatski
- **Pametno upozorenje o postojećim prijevodima** - Obavještava korisnika ako već postoji prevedena datoteka i nudi opciju zamjene
- **Podrška za različite formate** - Rad s .srt datotekama i automatsko generiranje izlaznih datoteka s ispravnim imenima
- **Obrada imena datoteka** - Automatsko prepoznavanje sezona i epizoda u nazivima datoteka
- **Integracija s video datotekama** - Povezivanje s odgovarajućim video datotekama (.mkv, .mp4, itd.)
- **Napredna obrada teksta** - Detekcija i obrada ispuna, kontekstualno prevođenje, prilagođavanje prijevoda temeljem konteksta

## Kako instalirati i pokrenuti

### Korištenje izvornog koda

1. Klonirajte repozitorij ili preuzmite izvorni kod
2. Kreirajte i aktivirajte virtualno okruženje (preporučeno):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Instalirajte potrebne pakete:

```powershell
pip install -r requirements.txt
```

4. Pokrenite aplikaciju:

```powershell
python improved_translator.py
```

### Korištenje izvršne datoteke (EXE)

1. Preuzmite najnovu verziju izvršne datoteke s izdanja (Releases)
2. Pokrenite `sinkronizator_titlova.exe`

## Kako koristiti

1. **Odabir ulazne datoteke** - Odaberite .srt datoteku za prevođenje ili mapu s više datoteka
2. **Postavke prijevoda** - Prilagodite postavke prema potrebama:
   - Uključite/isključite korištenje Google Gemini API-ja
   - Odaberite izlaznu lokaciju
   - Dodajte dodatne opcije ako je potrebno
3. **Pokrenite prevođenje** - Kliknite na gumb "Prevedi titlove"
4. **Pratite napredak** - Statusna traka će prikazivati napredak prevođenja

Aplikacija će vas obavijestiti ako već postoji prevedena datoteka na odredištu i pitati želite li je zamijeniti.

## Zahtjevi sustava

- Python 3.8 ili noviji
- Najmanje 4GB RAM-a (preporučeno 8GB+)
- Slobodan prostor na disku od najmanje 1GB za modele i privremene datoteke
- Internet veza za preuzimanje modela i korištenje Google Gemini API-ja

## Konfiguracija

### Postavke API ključeva

Za korištenje Google Gemini API-ja potrebno je postaviti odgovarajući API ključ u `.env` datoteci:

```bash
GOOGLE_API_KEY=vaš_google_api_ključ
```

### Korisnički rječnik

U datoteci `osnovni_rijecnik.txt` možete dodavati vlastite prijevode specifičnih izraza. Format je jednostavan:

```bash
engleska_riječ = hrvatski_prijevod
```

### Metapodaci

Za bolje prepoznavanje imena likova i drugih specifičnih pojmova, možete koristiti CSV datoteku s metapodacima. Primjer formata:

```csv
original,prijevod,rod,uloga
John,Marko,M,glavni lik
Sarah,Sandra,Ž,pomoćni lik
```

## Najnovije promjene (v0.12)

### Nove značajke
- Dodana obavijest o postojećim prevedenim datotekama
- Poboljšano korisničko sučelje s boljim prikazom napretka
- Automatsko prepoznavanje i obrada imena datoteka
- Podrška za povezivanje s video datotekama

### Poboljšanja
- Poboljšana stabilnost pri dugotrajnom radu
- Brže učitavanje modela
- Bolja podrška za različite formate datoteka

### Ispravci grešaka
- Ispravljen problem s kodiranjem znakova u nekim slučajevima
- Poboljšana kompatibilnost s različitim verzijama Windowsa
- Ispravljeni manji problemi s korisničkim sučeljem

## Rješavanje problema

### Česti problemi i rješenja

1. **Problem s učitavanjem modela**
   - Provjerite imate li dobru internetsku vezu za preuzimanje modela
   - Pokušajte ručno preuzeti model koristeći naredbu:
     ```powershell
     python -c "from transformers import MarianTokenizer, MarianMTModel; MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-tc-base-en-sh'); MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-tc-base-en-sh')"
     ```

2. **Greške s memorijom**
   - Pokušajte koristiti manji model
   - Zatvorite druge aplikacije koje troše puno memorije
   - Smanjite veličinu batch-a u postavkama

3. **Problemi s kodiranjem znakova**
   - Provjerite je li izvorna datoteka u UTF-8 formatu
   - Ako imate problema s prikazom hrvatskih znakova, provjerite postavke sustava

## Podrška i doprinosi

Ako naiđete na probleme ili imate prijedloge za poboljšanja, slobodno otvorite novi "issue" na GitHub repozitoriju.

## Licenca

Ovaj softver je dostupan pod MIT licencom. Pogledajte datoteku [LICENSE](LICENSE) za više informacija.

## Zahvale

- Timu iza Hugging Face Transformers biblioteke
- Razvijačima MarianMT modela
- Google-u za Gemini API

## Kontakt

Za sva pitanja i sugestije, možete me kontaktirati putem:
- E-mail: vaš.email@primjer.com
- Facebook: [Denis Sakač](https://www.facebook.com/sdenis.vr/)

---

© 2023-2024 Denis Sakač. Sva prava pridržana.
