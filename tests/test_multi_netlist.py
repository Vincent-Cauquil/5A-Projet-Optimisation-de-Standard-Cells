# tests/test_multi_netlist.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pool import SequentialPool
import pandas as pd

# Charger plusieurs netlists
pool = SequentialPool([
    "netlists/templates/rc_filter.cir",
    "netlists/templates/rc_gain.cir"
])

params = pd.DataFrame({
    "R_val": [100, 1000, 10000],
    "C_val": [1e-6, 1e-6, 1e-6],
})

results = pool.run(params)
print("\n" + "="*60)
print("Résultats consolidés:")
print(results)
