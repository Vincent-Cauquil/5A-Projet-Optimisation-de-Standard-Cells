# tests/test_rc_filter.py
from pathlib import Path
from pyngs.core import NGSpiceInstance
import pandas as pd
import sys

def test_rc_filter():
    # 1. Chemin basé sur TA CAPTURE D'ÉCRAN
    # Utilise .resolve() pour garantir un chemin absolu (obligatoire pour libngspice souvent)
    netlist_path = Path("netlists/templates/rc_filter.cir").resolve()
    
    print("--- DIAGNOSTIC ---")
    print(f"Chemin visé : {netlist_path}")
    
    # 2. Vérification de l'existence
    if not netlist_path.exists():
        print(f"❌ ERREUR : Le fichier n'existe pas à cet emplacement.")
        sys.exit(1)
        
    # 3. Vérification du contenu (pour être sûr qu'il n'est pas vide)
    try:
        content = netlist_path.read_text()
        if not content.strip():
            print("❌ ERREUR : Le fichier est vide !")
            sys.exit(1)
        print("✅ Fichier trouvé et non vide.")
        print(f"Début du fichier : {content[:50]}...") # Affiche les 50 premiers caractères
    except Exception as e:
        print(f"❌ ERREUR de lecture du fichier : {e}")
        sys.exit(1)
    print("------------------")

    # 4. Simulation
    configs = pd.DataFrame({
        "R_val": [1e3, 10e3, 1e3],
        "C_val": [1e-6, 100e-9, 100e-9]
    })
    
    results = []
    for idx, row in configs.iterrows():
        inst = NGSpiceInstance()
        
        # On passe le chemin absolu converti en string
        inst.load(str(netlist_path))
        
        inst.set_parameter("R_val", row["R_val"])
        inst.set_parameter("C_val", row["C_val"])
        inst.run()
        
        # Vérifie si 'fc' est bien mesuré
        try:
            fc = inst.get_measure('fc')
            results.append(fc)
        except Exception as e:
            print(f"Erreur mesure ligne {idx}: {e}")
            results.append(None)
            
        inst.stop()
    
    configs['fc_sim'] = results
    print("\nRÉSULTATS :")
    print(configs)

if __name__ == "__main__":
    test_rc_filter()