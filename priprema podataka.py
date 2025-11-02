import pandas as pd
import re

print("Pokretanje skripte za pripremu podataka (Koraci 1 i 2)...")

try:
    # --- Učitavanje CSV datoteka ---
    # Učitava CSV sa svim rečenicama
    df_script = pd.read_csv("Game_of_Thrones_Script.csv")
    # Učitava CSV s imenima i definiranim rodovima
    df_genders = pd.read_csv("imena_kategorizirana.csv")

    print("\n--- Učitana Skripta (Game_of_Thrones_Script.csv) ---")
    print(f"Pronađeni stupci: {df_script.columns.tolist()}")
    
    print("\n--- Učitani Rodovi (imena_kategorizirana.csv) ---")
    print(f"Pronađeni stupci: {df_genders.columns.tolist()}")

    # --- Korak 1 (djelomično): Spajanje podataka ---
    
    # Standardizacija imena stupca za spajanje
    # Preimenuj 'name' -> 'Name' i 'gender' -> 'Rod' u df_genders
    # da odgovara stupcu 'Name' u df_script
    df_genders.rename(columns={'name': 'Name', 'gender': 'Rod'}, inplace=True)
    
    print(f"\nPreimenovani stupci u df_genders za spajanje: {df_genders.columns.tolist()}")

    # Spajanje (Merge) dviju tablica na temelju stupca 'Name'
    # 'how='left'' osigurava da zadržimo sve rečenice iz skripte,
    # čak i ako za nekog lika nemamo definiran rod.
    df_merged = pd.merge(df_script, df_genders, on='Name', how='left')
    
    print("\n--- Spojeni podaci (Skripta + Rod) ---")
    print(df_merged.head())
    
    redaka_bez_roda = df_merged['Rod'].isnull().sum()
    print(f"\nBroj redaka bez pronađenog roda (ostat će 'NaN'): {redaka_bez_roda} od {len(df_merged)}")

    # --- Korak 2: Ispravak specifičnih grešaka (Laki dio) ---
    print("\n--- Pokrećem Korak 2: Ispravak specifičnih grešaka ---")
    
    # Rječnik ispravaka:
    # (?i) = case-insensitive (neovisno o velikim/malim slovima)
    # \b = granica riječi (word boundary), da ne mijenja npr. "ciljati"
    specificni_ispravci = {
        r'(?i)\b(osip)\b': 'brzopletost',
        r'(?i)\b(cilj)\b': 'štićenik',
        r'(?i)\b(Crone)\b': 'Starica',
        r'(?i)Blackwater Bay': 'Zaljev Crnovode', # Nema \b jer su dvije riječi
        r'(?i)\b(Lanister)\b': 'Lannister'
    }

    # Osiguravamo da je stupac 'Sentence' tipa string za .replace()
    df_merged['Sentence'] = df_merged['Sentence'].astype(str)
    # Stvaramo novi stupac za ispravljeni tekst
    df_merged['Ispravljena_Recenica_Korak2'] = df_merged['Sentence']
    
    # Primjena ispravaka pomoću regularnih izraza
    for pattern, replacement in specificni_ispravci.items():
        df_merged['Ispravljena_Recenica_Korak2'] = df_merged['Ispravljena_Recenica_Korak2'].replace(
            pattern, 
            replacement, 
            regex=True
        )

    # Provjera jesu li napravljene ikakve promjene
    changed_rows_count = (df_merged['Sentence'] != df_merged['Ispravljena_Recenica_Korak2']).sum()
    
    if changed_rows_count > 0:
        print(f"\nUkupno {changed_rows_count} redaka je ispravljeno u Koraku 2.")
    else:
        print("\nNisu pronađene greške iz rječnika 'specificni_ispravci'.")
    
    # --- Korak 4 (djelomično): Generiranje konačnog filea ---
    output_filename = "pripremljeni_podaci_za_korak3.csv"
    df_merged.to_csv(output_filename, index=False)
    
    print(f"\nDatoteka '{output_filename}' je uspješno spremljena.")
    print("Ova datoteka sadrži spojene podatke i spremna je za Korak 3 (LLM obrada).")

except FileNotFoundError as e:
    print(f"GREŠKA: Datoteka nije pronađena. Provjerite jesu li 'Game_of_Thrones_Script.csv' i 'imena_kategorizirana.csv' dostupne. Detalji: {e}")
except KeyError as e:
    print(f"GREŠKA: Nije pronađen stupac (KeyError). Vjerojatno 'Name', 'name' ili 'gender'. Detalji: {e}")
except Exception as e:
    print(f"Dogodila se neočekivana greška: {e}")
