# Povijest promjena (Changelog)

Sve važnije promjene u projektu bit će dokumentirane u ovoj datoteci.

## [Nadolazeće] - 2025-11-03

### Popravljeno
- **Poboljšano praćenje napretka prevođenja** - Dodana je potpuna podrška za praćenje napretka tokom prevođenja, uključujući:
  - Realno ažuriranje postotka napretka u korisničkom sučelju
  - Prikaz preostalog vremena za završetak prevođenja
  - Praćenje napretka i za pojedinačne datoteke i za batch prevođenje
  - Poboljšana responzivnost korisničkog sučelja tokom dugotrajnih operacija

## [1.0.1] - 2025-11-03

### Promijenjeno
- Promijenjen model za prijevod s `Helsinki-NLP/opus-mt-en-zls` na `Helsinki-NLP/opus-mt-tc-base-en-sh` jer prethodni model nije bio zadovoljavajućeg kvalitete. Novi model pruža bolje rezultate u prijevodu s engleskog na hrvatski.
- Dodan je eksplicitan prefiks `>>hrv<<` za svaki tekst prije prevođenja kako bi se osiguralo da model uvijek prevodi na hrvatski jezik, čime se sprječava mogućnost miješanja srodnih jezika (npr. srpski).

### Dodano
- [ ] Dodati funkcionalnost za praćenje promjena u kodu

## [1.0.0] - 2025-10-26

### Dodano
- Početna verzija projekta za sinkronizaciju titlova s engleskog na hrvatski
- Podrška za obradu SRT datoteka
## [1.0.0] - 2025-10-26

### Dodano
- Automatsko prepoznavanje i povezivanje s video datotekama
- Podrška za različite formate imenovanja epizoda (S02E03, 2x03, itd.)
- Integracija s Jellyfin-om kroz automatsko preimenovanje titlova
- Detaljnije poruke o greškama i statusu

### Promijenjeno
- Poboljšana robusnost pri pretraživanju datoteka
- Optimizirano pretraživanje u više direktorija
- Poboljšano prepoznavanje sličnosti imena datoteka

### Popravljeno
- Ispravljeno prepoznavanje video datoteka u različitim direktorijima
- Poboljšana kompatibilnost s različitim formatima imena serija i filmova

## [0.9.1] - 2025-10-25

### Dodano
- Kontekstualna obrada disklejmera u prijevodima
- Integracija s Google Gemini API-jem za pročišćavanje prijevoda
- Podrška za rodno osviješten prijevod

### Promijenjeno
- Poboljšana obrada disklejmera s obzirom na kontekst
- Optimizirane performanse prijevoda
- Poboljšano logiranje promjena

### Popravljeno
- Ispravljeni manji bugovi u prepoznavanju konteksta
- Poboljšana točnost prijevoda govornih fraza

## [0.9.0] - 2025-10-25

### Dodano
- Inicijalna implementacija prijevoditelja
- Osnovna obrada SRT formata
- Podrška za obradu više datoteka

### Napomene
Ova datoteka će se ažurirati kako projekt napreduje. Molimo pratite promjene i dodajte svoje izmjene u odgovarajuće sekcije.

---

Format ovog CHANGELOG-a je baziran na [Keep a Changelog](https://keepachangelog.com/hr/1.0.0/).
