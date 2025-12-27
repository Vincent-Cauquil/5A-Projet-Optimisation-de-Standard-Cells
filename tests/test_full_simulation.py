# tests/test_full_simulation.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from old.netlist_generator import (
    NetlistGenerator, SimulationConfig, TransitionTest
)
from src.simulation.spice_runner import SpiceRunner

print("="*60)
print("Test COMPLET: Génération + Simulation")
print("="*60)

# 1. Initialisation
pdk = PDKManager("sky130")

# ✨ Créer les netlists DIRECTEMENT dans ngspice/ (pas dans un sous-dossier)
netlist_dir = pdk.pdk_root / "libs.tech" / "ngspice"
netlist_dir.mkdir(exist_ok=True, parents=True)

generator = NetlistGenerator(pdk, output_dir=netlist_dir)
runner = SpiceRunner(pdk.pdk_root)

# 2. Configuration
config = SimulationConfig(
    vdd=1.8,
    temp=27,
    corner="tt",
    cload=10e-15,
    trise=100e-12,
    tfall=100e-12
)

# 3. Tests de transition pour XOR2
transitions = [
    TransitionTest(
        name="A rising, B=0 (X should rise)",
        input_signals={"A": "0→1", "B": "0"},
        measures=[
            ".meas tran delay_A_rise_B0 TRIG v(A) VAL='SUPPLY/2' RISE=1 TARG v(X) VAL='SUPPLY/2' RISE=1"
        ]
    ),
    TransitionTest(
        name="A falling, B=0 (X should fall)",
        input_signals={"A": "1→0", "B": "0"},
        measures=[
            ".meas tran delay_A_fall_B0 TRIG v(A) VAL='SUPPLY/2' FALL=1 TARG v(X) VAL='SUPPLY/2' FALL=1"
        ]
    ),
    TransitionTest(
        name="A rising, B=1 (X should fall)",
        input_signals={"A": "0→1", "B": "1"},
        measures=[
            ".meas tran delay_A_rise_B1 TRIG v(A) VAL='SUPPLY/2' RISE=1 TARG v(X) VAL='SUPPLY/2' FALL=1"
        ]
    ),
]

# 4. Génération de la netlist
print("\n" + "="*60)
print("ÉTAPE 1: Génération de la netlist")
print("="*60)
netlist_file = generator.generate_delay_netlist("xor2", config, transitions)

# 5. Exécution de la simulation
print("\n" + "="*60)
print("ÉTAPE 2: Exécution NGSpice")
print("="*60)
result = runner.run_simulation(netlist_file, verbose=True)

# 6. Analyse des résultats
print("\n" + "="*60)
print("RÉSULTATS FINAUX")
print("="*60)

if result['success']:
    print("✅ Simulation réussie !")

    if result['measures']:
        print("\n📊 Délais mesurés:")
        for name, value in result['measures'].items():
            if 'delay' in name:
                print(f"   • {name}: {value*1e12:.3f} ps")
    else:
        print("\n⚠️  Aucune mesure extraite (vérifiez les triggers)")
else:
    print("❌ Simulation échouée")
    print("\nErreurs:")
    for error in result['errors']:
        print(f"   • {error}")
