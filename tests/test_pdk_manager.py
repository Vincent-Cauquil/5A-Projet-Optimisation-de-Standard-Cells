# tests/test_pdk_final.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager

print("="*60)
print("Test PDKManager avec CDL")
print("="*60)

pdk = PDKManager("sky130")

# Liste des XOR
print("\n✓ Cellules XOR disponibles:")
xor_cells = pdk.list_available_cells("xor")
for cell in xor_cells[:10]:
    print(f"  • {cell}")

# Info sur XOR2
print("\n✓ Informations XOR2_1:")
info = pdk.get_cell_info("xor2")
print(f"  Nom: {info['name']}")
print(f"  Ports: {', '.join(info['ports'])}")
print(f"  Signaux: {', '.join(info['signal_pins'])}")
print(f"  Alim: {', '.join(info['power_pins'])}")

# Extraction
print("\n✓ Extraction XOR2:")
xor2_file = pdk.get_cell_spice("xor2")
print(f"  Fichier: {xor2_file}")

# Contenu
print("\n📄 Contenu (50 premières lignes):")
print(xor2_file.read_text().split('\n')[:50])
