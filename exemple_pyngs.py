from pathlib import Path
from pyngs.core import NGSpiceInstance
import math

# 1. Calcul théorique
def fc_theorique(R, C):
    return 1 / (2 * math.pi * R * C)

# 2. Tester plusieurs valeurs
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
    inst.load("model/rc_filter.cir")
    inst.set_parameter("R", R)
    inst.set_parameter("C", C)
    inst.run()
    
    fc_theo = fc_theorique(R, C)
    fc_sim = inst.get_measure('fc')
    
    print(f"R={R/1e3:5.1f}kΩ C={C*1e9:5.0f}nF | fc_theo={fc_theo:8.1f}Hz fc_sim={fc_sim:8.1f}Hz")
    
    inst.stop()
