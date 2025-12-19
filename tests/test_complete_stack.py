# tests/test_complete_stack.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.spice_runner import SpiceRunner

print("="*60)
print("Test: Stack complet avec sky130_fd_sc_hd.spice")
print("="*60)

pdk = PDKManager("sky130")
runner = SpiceRunner(pdk.pdk_root)

ngspice_dir = pdk.pdk_root / "libs.tech" / "ngspice"
test_netlist = ngspice_dir / "test_complete.spice"

# Utiliser directement le fichier SPICE de la bibliothèque
lib_spice = pdk.pdk_root / "libs.ref" / "sky130_fd_sc_hd" / "spice" / "sky130_fd_sc_hd.spice"

netlist_content = f"""* Test avec bibliothèque complète
.lib sky130.lib.spice tt

* Inclure TOUTE la bibliothèque de cellules
.include {lib_spice}

* Alimentation
.param SUPPLY=1.8
VVDD VPWR 0 DC {{SUPPLY}}
VVSS VGND 0 DC 0
VVPB VPB 0 DC {{SUPPLY}}
VVNB VNB 0 DC 0

* Instancier l'inverseur (avec les bons noms de ports)
XINV A Y VPWR VGND VPB VNB sky130_fd_sc_hd__inv_1

* Signal d'entrée
VA A 0 PULSE(0 {{SUPPLY}} 0 100p 100p 5n 10n)

* Charge de sortie
CL Y 0 10f

* Simulation
.tran 1p 20n

.control
run
print v(A) v(Y)
* Utiliser la valeur numérique directement (1.8/2 = 0.9)
meas tran tphl TRIG v(A) VAL=0.9 RISE=1 TARG v(Y) VAL=0.9 FALL=1
meas tran tplh TRIG v(A) VAL=0.9 FALL=1 TARG v(Y) VAL=0.9 RISE=1
quit
.endc

.end
"""

with open(test_netlist, 'w') as f:
    f.write(netlist_content)

print(f"✓ Netlist créée: {test_netlist.name}")
print(f"✓ Utilise: {lib_spice.name}")

result = runner.run_simulation(test_netlist, verbose=True)

if result['success']:
    print("\n✅ Simulation réussie !")
    if result['measures']:
        print("\n📊 Mesures extraites:")
        for name, value in result['measures'].items():
            print(f"   {name}: {value}")
else:
    print("\n❌ Problème")
    for error in result['errors']:
        print(f"   • {error}")
