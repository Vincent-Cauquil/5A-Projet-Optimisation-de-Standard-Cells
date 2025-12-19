import sys
from pathlib import Path

# Ajouter la racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pool import SequentialPool
import pandas as pd
from pathlib import Path

pool = SequentialPool(Path("netlists/templates/rc_filter.cir"))

params = pd.DataFrame({
    "R_val": [10, 100, 1000, 10000, 10, 100, 1000, 10000],
    "C_val": [1e-6, 1e-6, 1e-6, 1e-6, 2e-6, 2e-6, 2e-6, 2e-6],
})

results = pool.run(params)
print(results)
