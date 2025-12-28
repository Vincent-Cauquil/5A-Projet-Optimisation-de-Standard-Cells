from pathlib import Path
import re
from src.simulation.pdk_manager import PDKManager

def debug_extraction(cell_name="sky130_fd_sc_hd__inv_1"):
    print(f"🔍 DEBUG EXTRACTION pour : {cell_name}")
    
    # 1. Localiser le fichier
    pdk = PDKManager("sky130")
    lib_path = pdk.pdk_root / "libs.ref" / "sky130_fd_sc_hd" / "spice" / "sky130_fd_sc_hd.spice"
    
    print(f"📂 Fichier cible : {lib_path}")
    if not lib_path.exists():
        print("❌ ERREUR : Le fichier n'existe pas !")
        return

    # 2. Lire le fichier brut
    try:
        with open(lib_path, 'r') as f:
            lines = f.readlines()
        print(f"✅ Fichier lu ({len(lines)} lignes)")
    except Exception as e:
        print(f"❌ Erreur lecture : {e}")
        return

    # 3. Simulation de l'extraction ligne par ligne
    in_cell = False
    found_transistors = 0
    
    print("\n--- DÉBUT ANALYSE ---")
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Détection début cellule
        # On teste une correspondance plus souple
        if cell_name in line and ".subckt" in line.lower():
            print(f"🟢 [Ligne {i+1}] DÉBUT CELLULE TROUVÉ : '{line}'")
            in_cell = True
            continue
            
        if in_cell:
            # Fin de cellule
            if line.lower().startswith(".ends"):
                print(f"🔴 [Ligne {i+1}] FIN CELLULE : '{line}'")
                break
            
            # Détection transistor
            if line.upper().startswith('X') or line.upper().startswith('M'):
                print(f"  👉 [Ligne {i+1}] Transistor détecté : '{line}'")
                
                # Test du Regex de nettoyage
                match = re.search(r'(w|l)=([0-9\.\+\-eE]+)u', line, re.IGNORECASE)
                if match:
                    print(f"     ✅ Regex Match : {match.group(0)}")
                else:
                    print(f"     ⚠️  Regex Fail (Pas de 'u' ?) : '{line}'")
                
                found_transistors += 1
            else:
                # Affiche les lignes ignorées pour comprendre pourquoi
                if line:
                    print(f"  Resultat ignoré : '{line}'")

    print("\n--- RÉSULTAT ---")
    if found_transistors == 0:
        print("❌ AUCUN TRANSISTOR TROUVÉ. Vérifie la logique 'startswith(X/M)' ou le nom de la cellule.")
    else:
        print(f"✅ {found_transistors} transistors trouvés.")

if __name__ == "__main__":
    debug_extraction()