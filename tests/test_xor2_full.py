# tests/test_xor2_full.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.characterization import XOR2Characterization
import pandas as pd

# Paramètres de sweep
param_sweep = pd.DataFrame({
    "SUPPLY": [1.8, 1.8, 1.8, 1.65, 1.95],
    "TEMP": [27, -40, 125, 27, 27],
    "CLOAD": [10e-15, 10e-15, 10e-15, 50e-15, 10e-15],
    "FREQ": [100e6, 100e6, 100e6, 100e6, 50e6],
    "trise": [100e-12, 100e-12, 100e-12, 100e-12, 500e-12],
    "tfall": [100e-12, 100e-12, 100e-12, 100e-12, 500e-12]
})

# Caractérisation
char = XOR2Characterization()
results = char.run_full_characterization(param_sweep)

# Sauvegarder
char.save_results(results)

# Afficher résumé
print("\n" + "="*60)
print("RÉSUMÉ")
print("="*60)
print(results[['SUPPLY', 'TEMP', 'delay_avg', 'static_L_avg', 'PDP', 'EDP']].to_string())
