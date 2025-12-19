# tests/test_pdk_structure.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager

print("="*60)
print("Diagnostic structure PDK sky130")
print("="*60)

pdk = PDKManager("sky130")

print("\n📁 Fichiers importants:")
print(f"   • PDK root: {pdk.pdk_root}")

# Vérifier les modèles
print("\n🔍 Recherche des modèles de transistors:")
model_dir = pdk.pdk_root / "libs.ref" / "sky130_fd_pr" / "spice"
if model_dir.exists():
    print(f"   ✓ Répertoire modèles trouvé: {model_dir}")
    for file in model_dir.glob("*.spice"):
        print(f"     • {file.name}")
else:
    print(f"   ❌ Répertoire modèles introuvable")

# Vérifier la lib ngspice
print("\n🔍 Bibliothèque NGSpice:")
lib_file = pdk.pdk_root / "libs.tech" / "ngspice" / "sky130.lib.spice"
if lib_file.exists():
    print(f"   ✓ Trouvée: {lib_file}")
    # Lire les premières lignes
    with open(lib_file) as f:
        lines = f.readlines()[:20]
        print("\n   Premières lignes:")
        for line in lines:
            if line.strip():
                print(f"     {line.rstrip()}")
else:
    print(f"   ❌ Introuvable")

# Test des includes
print("\n📝 Includes générés:")
includes = pdk.get_complete_includes("tt")
print(includes)
