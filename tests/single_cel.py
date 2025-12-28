import sys
import os
import shutil
import re
from pathlib import Path

# Ajout du chemin src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig
from src.simulation.spice_runner import SpiceRunner

def debug_single_cell():
    CELL_NAME = "sky130_fd_sc_hd__inv_1"
    OUTPUT_FILE = Path("debug_inv_1.spice")
    
    print(f"🔬 --- DÉBUT DU TEST UNITAIRE : {CELL_NAME} ---")

    # 1. Setup
    try:
        pdk = PDKManager("sky130")
        gen = NetlistGenerator(pdk)
        runner = SpiceRunner(pdk.pdk_root)
        config = SimulationConfig() 
    except Exception as e:
        print(f"❌ Erreur d'initialisation : {e}")
        return

    # 2. Génération de la Netlist
    print("\n📝 1. Génération de la Netlist...")
    try:
        netlist_path = gen.generate_characterization_netlist(
            cell_name=CELL_NAME,
            output_path=str(OUTPUT_FILE),
            config=config
        )
        print(f"   ✅ Fichier généré : {netlist_path}")
    except Exception as e:
        print(f"❌ Erreur de génération : {e}")
        return

    # 3. Inspection du fichier (Vérification Physique)
    print("\n🔍 2. Inspection du contenu (Transistors)...")
    if not OUTPUT_FILE.exists():
        print("❌ Le fichier n'existe pas !")
        return
    
    print(OUTPUT_FILE)

    with open(OUTPUT_FILE, 'r') as f:
        content = f.read()
        
    lines = content.split('\n')
    has_error = False
    for line in lines:
        if line.strip().upper().startswith("X0") or line.strip().upper().startswith("M1"):
            print(f"   👉 Ligne trouvée : {line.strip()}")
            
            # Vérification des unités
            if "w=650000" in line:
                print("      ⚠️  ALERTE : Valeur géante détectée (650000) ! Problème de nettoyage.")
                has_error = True
            elif "w=0.65u" in line or "w=0.6500u" in line:
                print("      ✅ Unité correcte détectée (0.65u).")
            elif "scale=1e-6" in content and "w=0.65u" in line:
                 print("      ❌ ERREUR FATALE : .option scale + w=0.65u = Transistor microscopique !")
                 has_error = True

    # 4. Simulation NGSPICE
    print("\n🚀 3. Lancement de la Simulation...")
    result = runner.run_simulation(netlist_path, verbose=True)

    if result['success']:
        print("\n🎉 SUCCÈS SIMULATION !")
        print("📊 Mesures extraites :")
        for k, v in result['measures'].items():
            print(f"   - {k}: {v}")
    else:
        print("\n💀 ÉCHEC SIMULATION")
        
        # Affichage des erreurs brutes NGSPICE
        print("\n📜 --- LOG NGSPICE (STDERR) ---")
        if result.get('stderr'):
            print(result['stderr'])
        else:
            print("(Aucun stderr capturé)")
            
        print("\n📜 --- LOG NGSPICE (STDOUT - 20 dernières lignes) ---")
        if result.get('stdout'):
            print("\n".join(result['stdout'].split('\n')[-20:]))
        else:
            print("(Aucun stdout capturé)")
            
        # Tentative de lancement manuel pour voir l'erreur en direct
        print("\n🔧 --- TENTATIVE MANUELLE ---")
        ngspice_cmd = shutil.which("ngspice")
        cmd = f"{ngspice_cmd} -b {OUTPUT_FILE}"
        print(f"Exécution de : {cmd}")
        os.system(cmd)

if __name__ == "__main__":
    debug_single_cell()