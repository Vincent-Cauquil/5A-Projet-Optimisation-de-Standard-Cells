import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.spice_runner import SpiceRunner
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig

print("="*60)
print("Test: Générateur de netlists V2")
print("="*60)

pdk = PDKManager("sky130")
generator = NetlistGenerator(pdk)
runner = SpiceRunner(pdk.pdk_root)

# Test inverseur avec transitions par défaut
cell = "sky130_fd_sc_hd__inv_1"
config = SimulationConfig(vdd=1.8, cload=10e-15)

print(f"\n📝 Génération pour: {cell}")
netlist = generator.generate_delay_netlist(cell, config)
print(f"✓ Netlist: {netlist.name}")

print(f"\n🔄 Simulation...")
result = runner.run_simulation(netlist, verbose=False)

if result['success']:
    print(f"✅ Succès!")
    if result['measures']:
        print(f"\n📊 Mesures:")
        for name, value in result['measures'].items():
            print(f"   {name}: {value*1e12:.3f} ps")
else:
    print(f"❌ Échec")
    for err in result['errors']:
        print(f"   • {err}")