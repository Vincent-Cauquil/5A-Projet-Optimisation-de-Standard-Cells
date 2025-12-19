# tests/test_sequential_pool.py
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
