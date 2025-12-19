# tests/test_models_loading.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.spice_runner import SpiceRunner

print("="*60)
print("Test: Chargement des modèles PDK")
print("="*60)

pdk = PDKManager("sky130")
runner = SpiceRunner(pdk.pdk_root)

# Créer une netlist minimale dans ngspice/
ngspice_dir = pdk.pdk_root / "libs.tech" / "ngspice"
test_netlist = ngspice_dir / "test_models.spice"

netlist_content = """* Test chargement modèles
.lib sky130.lib.spice tt

* Simple NMOS
MN1 drain gate 0 0 nfet_01v8 W=1u L=0.15u
VGS gate 0 DC 1.8
VDS drain 0 DC 1.8

.dc VGS 0 1.8 0.1

.control
run
print all
quit
.endc

.end
"""

with open(test_netlist, 'w') as f:
    f.write(netlist_content)

print(f"✓ Netlist de test créée: {test_netlist.name}")

result = runner.run_simulation(test_netlist, verbose=True)

if result['success']:
    print("\n✅ Les modèles se chargent correctement !")
else:
    print("\n❌ Problème de chargement des modèles")
    for error in result['errors']:
        print(f"   • {error}")
