#!/usr/bin/env python3
"""Test simple de la cellule XOR2_1"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig
from src.simulation.spice_runner import SpiceRunner

def main():
    """Test de xor2_1"""
    
    print("=" * 80)
    print("🧪 TEST XOR2_1")
    print("=" * 80)
    
    # Initialisation
    pdk = PDKManager("sky130")
    gen = NetlistGenerator(pdk)
    runner = SpiceRunner(pdk.pdk_root)
    
    cell_name = "sky130_fd_sc_hd__xor2_1"
    
    # Configuration
    config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner="tt",
        cload=10e-15,
        trise=100e-12,
        tfall=100e-12
    )
    
    print(f"\n📋 Configuration:")
    print(f"   Cellule  : {cell_name}")
    print(f"   VDD      : {config.vdd} V")
    print(f"   Temp     : {config.temp} °C")
    print(f"   Corner   : {config.corner}")
    print(f"   C_load   : {config.cload*1e15:.1f} fF")
    print(f"   Slew     : {config.trise*1e12:.0f} ps")
    
    # Génération netlist
    print(f"\n{'─'*80}")
    print("📝 GÉNÉRATION NETLIST")
    print(f"{'─'*80}")
    
    try:
        netlist_path = gen.generate_netlist(cell_name, config)
        print(f"✅ Netlist générée: {netlist_path}")
        
        # Afficher la netlist
        print(f"\n{'─'*80}")
        print("📄 CONTENU NETLIST")
        print(f"{'─'*80}\n")
        
        with open(netlist_path, 'r') as f:
            content = f.read()
            print(content)
        
        # Simulation
        print(f"\n{'─'*80}")
        print("⚡ SIMULATION NGSPICE")
        print(f"{'─'*80}")


        result = runner.run_simulation(netlist_path, verbose=True)  # ← verbose=True pour voir la sortie

        # ✅ Vérifier le succès
        if not result['success']:
            print("\n❌ SIMULATION ÉCHOUÉE")
            if result['errors']:
                print("\n⚠️  Erreurs:")
                for error in result['errors']:
                    print(f"   • {error}")
            return 1

        # ✅ Extraire les mesures
        measures = result['measures']

        if measures:
            print("\n📊 RÉSULTATS:")
            print(f"{'─'*80}")

            # Séparer délais et consommation
            delays = {k: v for k, v in measures.items() if k.startswith(('tplh', 'tphl', 'delay'))}
            power = {k: v for k, v in measures.items() if k.startswith(('energy', 'power'))}

            if delays:
                print("\n⏱️  Délais:")
                for key, value in sorted(delays.items()):
                    print(f"   {key:25s} = {value*1e12:8.3f} ps")

            if power:
                print("\n⚡ Consommation:")

                # Énergie totale
                if 'energy_dyn' in power:
                    print(f"   {'Energy (total)':25s} = {power['energy_dyn']*1e15:8.3f} fJ")

                # Puissance moyenne
                if 'power_avg' in power:
                    print(f"   {'Power (avg)':25s} = {power['power_avg']*1e6:8.3f} µW")

                # Détails par test
                test_energies = {k: v for k, v in power.items() if k.startswith('energy_test')}
                if test_energies:
                    print("\n   Par transition:")
                    for key, value in sorted(test_energies.items()):
                        test_num = key.replace('energy_test', '')
                        print(f"      Test {test_num:2s} : {value*1e15:8.3f} fJ")

            # Afficher toutes les mesures brutes
            print("\n📋 Mesures brutes:")
            for key, value in sorted(measures.items()):
                print(f"   {key}: {value}")

        else:
            print("⚠️  Aucune mesure extraite")
            print("\n📤 Sortie NGSpice (dernières lignes):")
            print(result['stdout'][-2000:])  # Afficher les 2000 derniers caractères
            
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n{'='*80}")
    print("✅ TEST TERMINÉ")
    print(f"{'='*80}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
