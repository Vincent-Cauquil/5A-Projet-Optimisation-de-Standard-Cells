# tests/test_simulation_with_modification.py

"""
Test de simulation avec modification de cellule
Utilise SpiceRunner au lieu de pyngs directement
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig
from src.simulation.spice_runner import SpiceRunner
from src.optimization.cell_modifier import CellModifier

def print_section(title: str):
    """Affiche un séparateur de section"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}\n")


def print_measures(measures: dict, title: str = "MESURES"):
    """Affiche les mesures de manière formatée"""
    if not measures:
        print(f"⚠️  Aucune mesure dans {title}")
        return
    
    print(f"\n📊 {title}:")
    print(f"{'─'*80}")
    
    # Séparer par catégorie
    delays = {k: v for k, v in measures.items() if k.startswith(('tplh', 'tphl', 'delay'))}
    energies = {k: v for k, v in measures.items() if k.startswith('energy')}
    powers = {k: v for k, v in measures.items() if k.startswith('power')}
    
    if delays:
        print("\n⏱️  Délais:")
        for key, value in sorted(delays.items()):
            print(f"   {key:25s} = {value*1e12:8.3f} ps")
    
    if energies:
        print("\n⚡ Énergies:")
        for key, value in sorted(energies.items()):
            print(f"   {key:25s} = {value*1e15:8.3f} fJ")
    
    if powers:
        print("\n🔋 Puissances:")
        for key, value in sorted(powers.items()):
            print(f"   {key:25s} = {value*1e6:8.3f} µW")


def compare_measures(m1: dict, m2: dict, labels: tuple = ("Original", "Modifié")):
    """Compare deux ensembles de mesures"""
    print(f"\n📊 COMPARAISON DES RÉSULTATS")
    print(f"{'─'*80}\n")
    
    print(f"{'Mesure':<25} {labels[0]:>15} {labels[1]:>15} {'Delta':>15}")
    print(f"{'─'*25} {'─'*15} {'─'*15} {'─'*15}")
    
    all_keys = sorted(set(m1.keys()) | set(m2.keys()))
    
    for key in all_keys:
        if key in m1 and key in m2:
            v1, v2 = m1[key], m2[key]
            
            # Choisir l'unité appropriée
            if key.startswith(('tplh', 'tphl', 'delay')):
                v1_str = f"{v1*1e12:.3f} ps"
                v2_str = f"{v2*1e12:.3f} ps"
            elif key.startswith('energy'):
                v1_str = f"{v1*1e15:.3f} fJ"
                v2_str = f"{v2*1e15:.3f} fJ"
            elif key.startswith('power'):
                v1_str = f"{v1*1e6:.3f} µW"
                v2_str = f"{v2*1e6:.3f} µW"
            else:
                v1_str = f"{v1:.3e}"
                v2_str = f"{v2:.3e}"
            
            # Calculer le delta
            if v1 != 0:
                delta = ((v2 - v1) / v1) * 100
                delta_str = f"{delta:+.1f}%"
            else:
                delta_str = "N/A"
            
            print(f"{key:<25} {v1_str:>15} {v2_str:>15} {delta_str:>15}")


def main():
    """Test complet : génération → simulation → modification → re-simulation"""
    
    print_section("🧪 TEST DE SIMULATION AVEC MODIFICATION")
    
    # ===== CONFIGURATION =====
    print("📋 Configuration:")
    
    cell_name = "sky130_fd_sc_hd__inv_1"
    config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner="tt",
        cload=10e-15,
        trise=100e-12,
        tfall=100e-12
    )
    
    print(f"   Cellule  : {cell_name}")
    print(f"   VDD      : {config.vdd} V")
    print(f"   Temp     : {config.temp} °C")
    print(f"   Corner   : {config.corner}")
    print(f"   C_load   : {config.cload*1e15:.1f} fF")
    print(f"   Slew     : {config.trise*1e12:.0f} ps")
    
    # ===== INITIALISATION =====
    print_section("📦 INITIALISATION")
    
    pdk = PDKManager("sky130")
    gen = NetlistGenerator(pdk)
    runner = SpiceRunner(pdk.pdk_root)
    
    print(f"✅ PDK chargé: {pdk.pdk_root}")
    
    # ===== SIMULATION ORIGINALE =====
    print_section("🔷 ÉTAPE 1: SIMULATION ORIGINALE")
    
    try:
        # Générer netlist
        print("📝 Génération de la netlist originale...")
        netlist_orig_str = gen.generate_characterization_netlist(cell_name, config, output_path="/tmp/inv_orig.sp")
        netlist_orig = Path(netlist_orig_str)
        print(f"✅ Netlist: {netlist_orig}")
        
        # Simuler
        print("\n⚡ Simulation NGSpice...")
        result_orig = runner.run_simulation(netlist_orig, verbose=False)
        
        if not result_orig['success']:
            print("❌ Simulation originale échouée")
            for error in result_orig['errors']:
                print(f"   • {error}")
            return 1
        
        print("✅ Simulation réussie")
        
        measures_orig = result_orig['measures']
        print_measures(measures_orig, "RÉSULTATS ORIGINAUX")
        
    except Exception as e:
        print(f"❌ Erreur lors de la simulation originale: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== MODIFICATION =====
    print_section("🔷 ÉTAPE 2: MODIFICATION DES TRANSISTORS")
    
    try:
        # Charger la netlist
        print(f"🔧 Chargement de {netlist_orig}...")
        modifier = CellModifier(netlist_orig)
        
        # Afficher l'état initial
        widths_init = modifier.get_transistor_widths()
        print(f"\n📏 Largeurs initiales:")
        for name, width in widths_init.items():
            print(f"   {name}: {width:.1f} nm")
        
        # Modifier les largeurs
        print(f"\n🔧 Modification des largeurs:")
        new_widths = {
            'X0': 700.0,  # NFET: 650 → 700 nm
            'X1': 1200.0  # PFET: 1000 → 1200 nm
        }
        
        for name, new_width in new_widths.items():
            old_width = widths_init[name]
            modifier.modify_width(name, new_width)
            print(f"   {name}: {old_width:.1f} nm → {new_width:.1f} nm")
        
        # Sauvegarder
        netlist_mod_str = "/tmp/inv_modified.sp"
        modifier.apply_modifications(netlist_mod_str)
        netlist_mod = Path(netlist_mod_str) 
        print(f"\n✅ Netlist modifiée: {netlist_mod}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la modification: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== SIMULATION MODIFIÉE =====
    print_section("🔷 ÉTAPE 3: SIMULATION DE LA CELLULE MODIFIÉE")
    
    try:
        # Simuler
        print("⚡ Simulation NGSpice...")
        result_mod = runner.run_simulation(netlist_mod, verbose=False)
        
        if not result_mod['success']:
            print("❌ Simulation modifiée échouée")
            for error in result_mod['errors']:
                print(f"   • {error}")
            
            # Debug: comparer les netlists
            print("\n🔍 Comparaison des netlists:")
            print("\n--- ORIGINALE ---")
            with open(netlist_orig, 'r') as f:
                for line in f:
                    if line.strip().startswith('X'):
                        print(line.rstrip())
            
            print("\n--- MODIFIÉE ---")
            with open(netlist_mod, 'r') as f:
                for line in f:
                    if line.strip().startswith('X'):
                        print(line.rstrip())
            
            return 1
        
        print("✅ Simulation réussie")
        
        measures_mod = result_mod['measures']
        print_measures(measures_mod, "RÉSULTATS MODIFIÉS")
        
    except Exception as e:
        print(f"❌ Erreur lors de la simulation modifiée: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # ===== COMPARAISON =====
    print_section("🔷 ÉTAPE 4: ANALYSE COMPARATIVE")
    
    compare_measures(measures_orig, measures_mod)
    
    # ===== RÉSUMÉ =====
    print_section("✅ TEST TERMINÉ AVEC SUCCÈS")
    
    print("📁 Fichiers générés:")
    print(f"   • Original : {netlist_orig}")
    print(f"   • Modifié  : {netlist_mod}")
    
    print("\n🔧 Modifications appliquées:")
    for name, width in new_widths.items():
        print(f"   • {name}: {widths_init[name]:.1f} nm → {width:.1f} nm")
    
    print("\n📊 Impact sur les performances:")
    if 'tphl_t1' in measures_orig and 'tphl_t1' in measures_mod:
        delay_change = ((measures_mod['tphl_t1'] - measures_orig['tphl_t1']) 
                       / measures_orig['tphl_t1'] * 100)
        print(f"   • Délai: {delay_change:+.1f}%")
    
    if 'energy_dyn' in measures_orig and 'energy_dyn' in measures_mod:
        energy_change = ((measures_mod['energy_dyn'] - measures_orig['energy_dyn']) 
                        / measures_orig['energy_dyn'] * 100)
        print(f"   • Énergie: {energy_change:+.1f}%")
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
