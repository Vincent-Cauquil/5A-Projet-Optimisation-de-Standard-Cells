# tests/test_netlist_generator.py (même fichier, relancez-le)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import (
    NetlistGenerator, SimulationConfig, TransitionTest
)

print("="*60)
print("Test NetlistGenerator CORRIGÉ")
print("="*60)

# Initialisation
pdk = PDKManager("sky130")
generator = NetlistGenerator(pdk)

# Configuration
config = SimulationConfig(
    vdd=1.8,
    temp=27,
    corner="tt",
    cload=10e-15,
    trise=100e-12,
    tfall=100e-12
)

# Tests de transition pour XOR2
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

# Génération
netlist_file = generator.generate_delay_netlist("xor2", config, transitions)

print(f"\n✓ Netlist créée: {netlist_file}")
print(f"\n📄 Contenu (affichage tronqué):\n")

content = netlist_file.read_text()
lines = content.split('\n')

# Afficher sections importantes
print("="*60)
print("SOURCES PWL:")
print("="*60)
for line in lines:
    if line.startswith('VA') or line.startswith('VB'):
        print(line)

print("\n" + "="*60)
print("MESURES:")
print("="*60)
for line in lines:
    if '.meas' in line:
        print(line)

print(f"\n📊 Durée totale simulation: {len(transitions) * 25}ns")
print(f"   • Test 1: 0-20ns")
print(f"   • Test 2: 25-45ns")
print(f"   • Test 3: 50-70ns")
