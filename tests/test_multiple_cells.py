#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.spice_runner import SpiceRunner
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig

def test_cell(generator, runner, cell_name, config):
    """Test une cellule et retourne les résultats"""
    print(f"\n{'='*60}")
    print(f"📝 Test: {cell_name}")
    print(f"{'='*60}")
    
    try:
        netlist = generator.generate_delay_netlist(cell_name, config)
        print(f"✓ Netlist générée: {netlist.name}")
        
        result = runner.run_simulation(netlist, verbose=False)
        
        if result['success'] and result['measures']:
            print(f"✅ Succès!")
            print(f"\n📊 Mesures de délai:")
            
            delays = {}
            for name, value in result['measures'].items():
                delay_ps = value * 1e12
                delays[name] = delay_ps
                print(f"   • {name:20s}: {delay_ps:8.3f} ps")
            
            return {'success': True, 'delays': delays}
        else:
            print(f"❌ Échec de simulation")
            if result['errors']:
                print("Erreurs:")
                for err in result['errors'][:3]:
                    print(f"   • {err}")
            return {'success': False, 'delays': {}}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {'success': False, 'delays': {}}

def main():
    print("="*60)
    print("Test de caractérisation multi-cellules")
    print("="*60)
    
    # Initialisation
    pdk = PDKManager("sky130")
    generator = NetlistGenerator(pdk)
    runner = SpiceRunner(pdk.pdk_root)
    
    # Configuration de simulation
    config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner="tt",
        cload=10e-15,
        trise=100e-12,
        tfall=100e-12
    )
    
    # Liste de cellules à tester
    test_cells = [
        "sky130_fd_sc_hd__inv_1",
        "sky130_fd_sc_hd__inv_2",
        "sky130_fd_sc_hd__buf_1",
        "sky130_fd_sc_hd__nand2_1",
        "sky130_fd_sc_hd__nor2_1",
        "sky130_fd_sc_hd__and2_1",
        "sky130_fd_sc_hd__or2_1",
        "sky130_fd_sc_hd__xor2_1",
    ]
    
    # Tester chaque cellule
    results = {}
    for cell in test_cells:
        result = test_cell(generator, runner, cell, config)
        results[cell] = result
    
    # Résumé
    print(f"\n{'='*60}")
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print(f"{'='*60}\n")
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    
    print(f"Cellules testées: {total_count}")
    print(f"Succès: {success_count}")
    print(f"Échecs: {total_count - success_count}")
    
    print(f"\n{'Cellule':<30} {'tphl (ps)':<15} {'tplh (ps)':<15}")
    print("-" * 60)
    
    for cell, result in results.items():
        if result['success']:
            delays = result['delays']
            tphl = delays.get('tphl', delays.get('delay_fall', 0))
            tplh = delays.get('tplh', delays.get('delay_rise', 0))
            cell_short = cell.replace('sky130_fd_sc_hd__', '')
            print(f"{cell_short:<30} {tphl:>12.2f}    {tplh:>12.2f}")
        else:
            cell_short = cell.replace('sky130_fd_sc_hd__', '')
            print(f"{cell_short:<30} {'ÉCHEC':<15}")
    
    print("\n" + "="*60)
    print("✅ Tests terminés")
    print("="*60)

if __name__ == "__main__":
    main()
