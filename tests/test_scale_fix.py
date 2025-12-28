import os
import subprocess
import shutil
import re
from pathlib import Path
import sys

# Ajout du root au path pour pouvoir importer src
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==============================================================================
# TEST 1 : VÉRIFICATION DE L'ÉCHELLE PICOMÈTRE (NGSPICE)
# ==============================================================================
def test_ngspice_physics_picometer():
    print("\n" + "="*60)
    print("--- TEST 1 : Validation Physique NGSPICE (Scale=1E-12) ---")
    print("="*60)
    print("Objectif : Vérifier que w=650000 + scale=1p est bien lu comme 0.65µm.")
    
    spice_filename = "test_pico_physics.spice"
    
    # On crée une netlist avec votre stratégie actuelle :
    # 1. Option scale = 1E-12 (picomètres)
    # 2. Valeurs entières brutes (ex: 650000) SANS unité
    netlist_content = """
* Test Picometer Scale Strategy
.option reltol=1e-3

* ICI L'OPTION CRITIQUE
.options parser scale=1E-12

* Modèle générique pour le test
.model nmos_test nmos level=1

* Le transistor tel qu'il sera écrit par votre nouveau code :
* w=650000 (sans 'u') -> 650000 * 1e-12 = 0.65e-6 mètres
M1 d g 0 0 nmos_test w=650000 l=150000

.control
op
echo "RESULTATS:"
* On imprime la largeur réelle comprise par le simulateur
print @m1[w]
quit
.endc
.end
"""
    
    with open(spice_filename, "w") as f:
        f.write(netlist_content)

    ngspice_cmd = shutil.which("ngspice")
    if not ngspice_cmd:
        print("❌ ERREUR : NGSPICE introuvable dans le PATH.")
        return

    try:
        result = subprocess.run(
            [ngspice_cmd, "-b", spice_filename], 
            capture_output=True, 
            text=True
        )
        
        # On cherche la valeur de W dans la sortie standard
        match = re.search(r"@m1\[w\]\s*=\s*([0-9\.eE\+\-]+)", result.stdout)

        if match:
            w_val = float(match.group(1))
            print(f"   • Entrée Netlist : w=650000, scale=1E-12")
            print(f"   • Sortie NGSPICE : {w_val:.2e} mètres")
            
            # On attend 0.65um = 6.5e-7 mètres
            target = 6.5e-7
            
            if abs(w_val - target) < 1e-9:
                print(f"✅ SUCCÈS PHYSIQUE : NGSPICE a bien calculé {target*1e6} µm.")
            else:
                print(f"❌ ÉCHEC PHYSIQUE : Valeur inattendue ({w_val}).")
                print(f"   Attendu : {target}")
        else:
            print("⚠️  Impossible de lire la sortie NGSPICE.")
            print("   Sortie brute :")
            print(result.stdout)

    finally:
        if os.path.exists(spice_filename):
            os.remove(spice_filename)

# ==============================================================================
# TEST 2 : VÉRIFICATION DU NETTOYAGE PYTHON (SUPPRESSION DU 'U')
# ==============================================================================
def test_generator_cleaning_logic():
    print("\n" + "="*60)
    print("--- TEST 2 : Validation du Nettoyage Python (Regex) ---")
    print("="*60)
    print("Objectif : Vérifier que le code Python transforme 'w=650000u' en 'w=650000'.")
    
    try:
        from src.simulation.netlist_generator import NetlistGenerator
        
        # 1. Setup : Création d'un faux environnement PDK temporaire
        tmp_dir = Path("./tmp_test_gen_pico")
        pdk_root = tmp_dir / "pdk"
        out_dir = tmp_dir / "output"
        
        # Nettoyage préventif
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)
        
        # Structure de dossiers minimale
        (pdk_root / "libs.tech/ngspice").mkdir(parents=True)
        (pdk_root / "libs.ref/sky130_fd_sc_hd/spice").mkdir(parents=True)
        
        # 2. Création d'un fichier source "Sale" (avec le 'u' problématique)
        fake_lib = pdk_root / "libs.ref/sky130_fd_sc_hd/spice/sky130_fd_sc_hd.spice"
        with open(fake_lib, "w") as f:
            f.write("* Fake Library for Unit Test\n")
            f.write(".subckt sky130_fd_sc_hd__inv_1 A Y VPWR VGND VPB VNB\n")
            # Voici la ligne typique extraite du PDK SkyWater
            f.write("X0 Y A VGND VNB sky130_fd_pr__nfet_01v8 w=650000u l=150000u\n")
            f.write(".ends\n")

        # Mock de l'objet PDKManager (juste besoin de pdk_root)
        class MockPDK:
            pass
        pdk = MockPDK()
        pdk.pdk_root = pdk_root

        # 3. Exécution du générateur
        # On utilise verbose=False pour ne pas polluer la console
        gen = NetlistGenerator(pdk, output_dir=out_dir, verbose=False)
        
        # On appelle la méthode qui fait le travail
        # Note : generate_characterization_netlist appelle _extract_transistors_from_cell
        output_file = gen.generate_characterization_netlist(
            "sky130_fd_sc_hd__inv_1", 
            str(out_dir / "test_clean.spice")
        )

        # 4. Vérification du résultat dans le fichier généré
        with open(output_file, "r") as f:
            content = f.read()

        print("\n🔍 Contenu généré (Lignes pertinentes) :")
        target_lines = []
        for line in content.splitlines():
            # On cherche les lignes X0 (transistor) ou .option
            if "X0" in line or "scale=" in line:
                print(f"   > {line.strip()}")
                target_lines.append(line.strip())

        # --- ASSERTIONS ---
        errors = []
        full_text = "\n".join(target_lines)
        
        # Check A: L'option scale est-elle présente ?
        if "scale=1E-12" in full_text or "scale=1e-12" in full_text:
            print("   ✅ Option scale=1E-12 trouvée.")
        else:
            errors.append("ERREUR : La ligne '.options parser scale=1E-12' est manquante !")
        
        # Check B: Le 'u' a-t-il disparu ?
        if "w=650000u" in full_text:
            errors.append("ERREUR : Le suffixe 'u' est encore présent (w=650000u).")
        elif "w=650000" in full_text:
            print("   ✅ Valeur nettoyée trouvée : w=650000 (sans unité).")
        else:
            errors.append("ERREUR : Impossible de trouver 'w=650000' dans le fichier.")

        if not errors:
            print("\n🎉 SUCCÈS TOTAL : Le générateur est prêt pour la simulation Picomètre.")
        else:
            print("\n❌ ÉCHEC DU TEST GÉNÉRATEUR :")
            for e in errors:
                print(f"   - {e}")

        # Nettoyage final
        shutil.rmtree(tmp_dir)

    except ImportError:
        print("❌ Impossible d'importer src.simulation.netlist_generator")
        print("   Assurez-vous d'être à la racine du projet.")
    except Exception as e:
        print(f"❌ Exception inattendue : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ngspice_physics_picometer()
    test_generator_cleaning_logic()