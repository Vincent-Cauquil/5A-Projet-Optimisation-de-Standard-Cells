from pathlib import Path
from pyngs.core import NGSpiceInstance
netlist_path = Path("inv.cir")
inst = NGSpiceInstance()
# Load netlist
inst.load(netlist_path)
# Run simulation
inst.run()
# List parameters & measures
print(inst.list_parameters())
print(inst.list_measures())
# Get measure value
print(f"delay_arzf = {inst.get_measure('delay_arzf')}")
# Set parameter value
vdd = 0.9
inst.set_parameter("vdd", vdd)
# Run a new simulation
inst.run()
# Get new delay value
print(f"delay_arzf (vdd={vdd}) = {inst.get_measure('delay_arzf')}")
# Stop simulator
inst.stop()