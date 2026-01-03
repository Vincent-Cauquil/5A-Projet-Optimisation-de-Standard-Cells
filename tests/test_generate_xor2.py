#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig

# Config
pdk = PDKManager("sky130")
gen = NetlistGenerator(pdk)
config = SimulationConfig(vdd=1.8, temp=27, corner="tt", cload=10e-15, trise=100e-12, tfall=100e-12)

# Générer
cell = "sky130_fd_sc_hd__xor2_1"
netlist_path = gen.generate_netlist(cell, config)

# Afficher
print(open(netlist_path).read())
