# tests/test_rc_filter.py
from pathlib import Path
from pyngs.core import NGSpiceInstance
import pandas as pd

def test_rc_filter():
    netlist = Path("netlists/templates/rc_filter.cir")
    
    # Tester plusieurs valeurs
    configs = pd.DataFrame({
        "R_val": [1e3, 10e3, 1e3],
        "C_val": [1e-6, 100e-9, 100e-9]
    })
    
    results = []
    for idx, row in configs.iterrows():
        inst = NGSpiceInstance()
        inst.load(str(netlist))
        inst.set_parameter("R_val", row["R_val"])
        inst.set_parameter("C_val", row["C_val"])
        inst.run()
        
        fc = inst.get_measure('fc')
        results.append(fc)
        inst.stop()
    
    configs['fc_sim'] = results
    print(configs)

if __name__ == "__main__":
    test_rc_filter()
