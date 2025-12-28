# tests/test_parse_log.py
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pool import SequentialPool
from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import SimulationConfig

# Sim unique
pdk = PDKManager("sky130", verbose=False)
config = SimulationConfig(vdd=1.8, temp=27)

pool = SequentialPool(pdk, config, verbose=True)

netlist = Path("benchmark_sims/test_inv_0000.cir")
result = pool._run_single_simulation(netlist)

print("\n📊 Résultat parsé:")
print(result)
