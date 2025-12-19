# tests/test_xor2.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pool import SequentialPool
import pandas as pd

# Charger les netlists XOR2
pool = SequentialPool([
    "netlists/templates/xor2/delay.cir",
    "netlists/templates/xor2/static.cir",
    "netlists/templates/xor2/energy.cir"
])

# Paramètres à explorer
params = pd.DataFrame({
    "VDD": [1.6, 1.8, 2.0],
    "CL": [5e-15, 10e-15, 20e-15],
})

results = pool.run(params)
print("\n" + "="*60)
print("Résultats XOR2:")
print(results)
