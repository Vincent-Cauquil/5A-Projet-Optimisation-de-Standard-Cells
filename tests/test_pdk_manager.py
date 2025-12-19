# tests/test_pdk_manager.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager

pdk = PDKManager("sky130")

print("\n=== Informations PDK ===")
print(f"Racine: {pdk.pdk_root}")
print(f"Librairie: {pdk.lib_path}")
print(f"Include: {pdk.get_lib_include('tt')}")

print("\n=== Cellules XOR ===")
cells = pdk.list_available_cells("xor")
for cell in cells[:5]:
    print(f"  - {cell}")

print("\n=== Fichier SPICE XOR2 ===")
xor2_path = pdk.get_cell_spice("xor2")
print(f"Path: {xor2_path}")
