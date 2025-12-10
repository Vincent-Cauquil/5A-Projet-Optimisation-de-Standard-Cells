from pathlib import Path
from pyngs.core import NGSpiceInstance
import math

# 1. Créer le fichier SPICE
netlist = """RC Low-Pass Filter

.param R=1k
.param C=1u

Vin in 0 DC 0 AC 1
R1 in out {R}
C1 out 0 {C}

.ac dec 100 10 1MEG
.measure ac fc WHEN vdb(out)=-3

.end
"""

Path("rc_filter.cir").write_text(netlist)
print("Fichier rc_filter.cir créé ✓")

# 2. Calcul théorique
def fc_theorique(R, C):
    return 1 / (2 * math.pi * R * C)

# 3. Test avec valeurs initiales
R = 1e3   # 1kΩ
C = 1e-6  # 1µF

print(f"\nR = {R/1e3} kΩ, C = {C*1e6} µF")
print(f"fc théorique = {fc_theorique(R, C):.2f} Hz")

# 4. Simulation NGSpice
inst = NGSpiceInstance()
inst.load("rc_filter.cir")
inst.set_parameter("R", R)
inst.set_parameter("C", C)
inst.run()

fc_sim = inst.get_measure('fc')
print(f"fc simulée   = {fc_sim:.2f} Hz")

inst.stop()

# 5. Tester plusieurs valeurs
print("\n" + "="*50)
print("Test avec plusieurs valeurs:")
print("="*50)

configs = [
    (1e3, 1e-6),    # 1kΩ, 1µF
    (10e3, 100e-9), # 10kΩ, 100nF
    (1e3, 100e-9),  # 1kΩ, 100nF
]

for R, C in configs:
    inst = NGSpiceInstance()
    inst.load("rc_filter.cir")
    inst.set_parameter("R", R)
    inst.set_parameter("C", C)
    inst.run()
    
    fc_theo = fc_theorique(R, C)
    fc_sim = inst.get_measure('fc')
    
    print(f"R={R/1e3:5.1f}kΩ C={C*1e9:5.0f}nF | fc_theo={fc_theo:8.1f}Hz fc_sim={fc_sim:8.1f}Hz")
    
    inst.stop()
