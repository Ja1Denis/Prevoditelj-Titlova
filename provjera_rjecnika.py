import os
from pathlib import Path

def provjeri_rječnike():
    # Definiraj moguće lokacije rječnika
    moguce_lokacije = [
        "osnovni_riječnik.txt",  # U trenutnom direktoriju
        "dictionary/osnovni_riječnik.txt",  # U poddirektoriju dictionary
        "../osnovni_riječnik.txt"  # U roditeljskom direktoriju
    ]
    
    for lokacija in moguce_lokacije:
        try:
            print(f"\nPokušavam pronaći rječnik na: {lokacija}")
            with open(lokacija, 'r', encoding='utf-8') as f:
                print(f"\nPronađen rječnik na: {os.path.abspath(lokacija)}")
                print(f"Veličina datoteke: {os.path.getsize(lokacija)} bajtova")
                print("-" * 80)
                
                # Pročitaj i prikaži prvih 20 redaka
                print("Prvih 20 redaka rječnika:")
                for i, line in enumerate(f):
                    if i < 20:
                        print(f"{i+1:2d}: {line.strip()}")
                    else:
                        break
                return True
                
        except FileNotFoundError:
            print(f"Nema datoteke na: {lokacija}")
        except Exception as e:
            print(f"Greška pri čitanju {lokacija}: {e}")
    
    print("\nNisam uspio pronaći datoteku rječnika na poznatim lokacijama.")
    print("Molim vas da:")
    print("1. Kopirate datoteku 'osnovni_riječnik.txt' u direktorij:")
    print(f"   {os.getcwd()}")
    print("2. Pokrenite skriptu ponovno")
    return False

# Pokretanje provjere
if __name__ == "__main__":
    provjeri_rječnike()
