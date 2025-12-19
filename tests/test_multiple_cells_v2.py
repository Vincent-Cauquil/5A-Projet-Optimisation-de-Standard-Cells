#!/usr/bin/env python3
import sys
from pathlib import Path
import re
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import time
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.spice_runner import SpiceRunner
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig

def extract_all_cells(spice_lib_path: Path) -> list:
    """Extrait tous les noms de cellules du fichier SPICE"""
    cells = []
    
    with open(spice_lib_path, 'r') as f:
        for line in f:
            line = line.strip()
            match = re.match(r'^\.subckt\s+(\S+)', line, re.IGNORECASE)
            if match:
                cell_name = match.group(1)
                if cell_name.startswith('sky130_fd_sc_hd__'):
                    cells.append(cell_name)
    
    return sorted(set(cells))

def categorize_cells(cells: list) -> dict:
    """Catégorise les cellules par type"""
    categories = {
        'inverters': [],
        'buffers': [],
        'nand': [],
        'nor': [],
        'and': [],
        'or': [],
        'xor': [],
        'xnor': [],
        'mux': [],
        'latches': [],
        'flip_flops': [],
        'other': []
    }
    
    for cell in cells:
        cell_lower = cell.lower()
        
        if '__inv_' in cell_lower:
            categories['inverters'].append(cell)
        elif '__buf_' in cell_lower or '__clkbuf_' in cell_lower:
            categories['buffers'].append(cell)
        elif '__nand' in cell_lower:
            categories['nand'].append(cell)
        elif '__nor' in cell_lower:
            categories['nor'].append(cell)
        elif '__and' in cell_lower:
            categories['and'].append(cell)
        elif '__xnor' in cell_lower:
            categories['xnor'].append(cell)
        elif '__xor' in cell_lower:
            categories['xor'].append(cell)
        elif '__or' in cell_lower:
            categories['or'].append(cell)
        elif '__mux' in cell_lower:
            categories['mux'].append(cell)
        elif '__dlx' in cell_lower or '__latch' in cell_lower:
            categories['latches'].append(cell)
        elif '__df' in cell_lower or '__sdff' in cell_lower:
            categories['flip_flops'].append(cell)
        else:
            categories['other'].append(cell)
    
    return {k: v for k, v in categories.items() if v}

def test_cell_worker(args):
    """Worker function pour tester une cellule (doit être picklable)"""
    cell_name, pdk_name, config_dict, counter, lock, total = args
    
    try:
        # Chaque worker crée ses propres instances
        pdk = PDKManager(pdk_name)
        generator = NetlistGenerator(pdk)
        runner = SpiceRunner(pdk.pdk_root)
        
        # Reconstruire la config
        config = SimulationConfig(**config_dict)
        
        # Générer et simuler
        netlist = generator.generate_delay_netlist(cell_name, config)
        result = runner.run_simulation(netlist, verbose=False)

        # Mise à jour du compteur (thread-safe)
        with lock:
            counter.value += 1
            current = counter.value
            cell_short = cell_name.replace('sky130_fd_sc_hd__', '')
            print(f"\r[{current}/{total}] Testing: {cell_short:<35}", end='', flush=True)

        if result['success'] and result['measures']:
            delays = {name: value * 1e12 for name, value in result['measures'].items()}
            return (cell_name, {'success': True, 'delays': delays, 'error': None})
        else:
            error_msg = result['errors'][0] if result['errors'] else "Unknown error"
            return (cell_name, {'success': False, 'delays': {}, 'error': error_msg})

    except Exception as e:
        with lock:
            counter.value += 1
        return (cell_name, {'success': False, 'delays': {}, 'error': str(e)})

def print_category_results(category_name: str, cells: list, results: dict):
    """Affiche les résultats d'une catégorie"""
    print(f"\n{'='*80}")
    print(f"📁 {category_name.upper()} ({len(cells)} cellules)")
    print(f"{'='*80}")
    
    success_count = sum(1 for c in cells if results.get(c, {}).get('success', False))
    print(f"✅ Succès: {success_count}/{len(cells)}")
    
    if success_count > 0:
        print(f"\n{'Cellule':<35} {'tphl (ps)':<15} {'tplh (ps)':<15} {'Moy (ps)':<15}")
        print("-" * 80)
        
        for cell in cells:
            result = results.get(cell, {})
            if result.get('success'):
                delays = result['delays']
                
                tphl_values = [v for k, v in delays.items() if 'tphl' in k.lower()]
                tplh_values = [v for k, v in delays.items() if 'tplh' in k.lower()]
                
                tphl = sum(tphl_values) / len(tphl_values) if tphl_values else 0
                tplh = sum(tplh_values) / len(tplh_values) if tplh_values else 0
                avg = (tphl + tplh) / 2 if (tphl or tplh) else 0
                
                cell_short = cell.replace('sky130_fd_sc_hd__', '')
                print(f"{cell_short:<35} {tphl:>12.2f}    {tplh:>12.2f}    {avg:>12.2f}")
    
    failed_cells = [c for c in cells if not results.get(c, {}).get('success', False)]
    if failed_cells:
        print(f"\n❌ Échecs ({len(failed_cells)}):")
        for cell in failed_cells[:5]:
            cell_short = cell.replace('sky130_fd_sc_hd__', '')
            error = results.get(cell, {}).get('error', 'Unknown')
            print(f"   • {cell_short}: {error[:60]}")
        if len(failed_cells) > 5:
            print(f"   ... et {len(failed_cells) - 5} autres")

def main():
    print("="*80)
    print("🔬 TEST PARALLÈLE DE TOUTES LES CELLULES STANDARD - SKY130 PDK")
    print("="*80)

    # Initialisation
    print("\n🔍 Initialisation du PDK...")
    pdk = PDKManager("sky130")
    
    # Détection du nombre de cœurs
    n_cores = cpu_count()
    n_workers = max(1, n_cores - 1)  # Laisser 1 cœur libre
    print(f"🚀 Utilisation de {n_workers} workers (sur {n_cores} cœurs disponibles)")

    # Configuration
    config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner="tt",
        cload=10e-15,
        trise=100e-12,
        tfall=100e-12
    )
    
    # Convertir config en dict pour le multiprocessing
    config_dict = {
        'vdd': config.vdd,
        'temp': config.temp,
        'corner': config.corner,
        'cload': config.cload,
        'trise': config.trise,
        'tfall': config.tfall
    }

    # Extraction de toutes les cellules
    print("📚 Extraction des cellules du fichier SPICE...")
    spice_lib = pdk.pdk_root / "libs.ref" / "sky130_fd_sc_hd" / "spice" / "sky130_fd_sc_hd.spice"
    all_cells = extract_all_cells(spice_lib)
    print(f"✓ {len(all_cells)} cellules trouvées")

    # Catégorisation
    print("🗂️  Catégorisation des cellules...")
    categories = categorize_cells(all_cells)
    
    print("\n📊 Répartition par catégorie:")
    for cat, cells in categories.items():
        print(f"   • {cat:<15}: {len(cells):>4} cellules")

    # Filtrer les catégories supportées
    supported_categories = ['inverters', 'buffers', 'nand', 'nor', 'and', 'or', 'xor', 'xnor']
    cells_to_test = []
    for cat in supported_categories:
        if cat in categories:
            cells_to_test.extend(categories[cat])
    
    print(f"\n🎯 Cellules supportées à tester: {len(cells_to_test)}")
    
    # Demander confirmation
    response = input("\n⚠️  Voulez-vous tester toutes ces cellules? (o/n) [o]: ").lower()
    if response and response not in ['o', 'y', 'yes', 'oui']:
        print("❌ Test annulé")
        return

    # Test parallèle de toutes les cellules
    print("\n" + "="*80)
    print(f"🚀 DÉMARRAGE DES TESTS PARALLÈLES ({n_workers} workers)")
    print("="*80)
    
    # Manager pour partager le compteur entre processus
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    total = len(cells_to_test)
    
    # Préparer les arguments pour chaque cellule
    pdk_name = "sky130"
    task_args = [(cell, pdk_name, config_dict, counter, lock, total) for cell in cells_to_test]
    
    # Lancer le pool de workers
    start_time = time.time()
    
    with Pool(processes=n_workers) as pool:
        results_list = pool.map(test_cell_worker, task_args)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Convertir la liste de résultats en dictionnaire
    results = dict(results_list)
    
    print("\n")  # Nouvelle ligne après la barre de progression

    # Affichage des résultats par catégorie
    for category in supported_categories:
        if category in categories:
            print_category_results(category, categories[category], results)

    # Résumé global
    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ GLOBAL")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)
    success_rate = (success_count / total_count * 100) if total_count > 0 else 0
    
    print(f"\n✅ Cellules testées: {total_count}")
    print(f"✅ Succès: {success_count} ({success_rate:.1f}%)")
    print(f"❌ Échecs: {total_count - success_count} ({100-success_rate:.1f}%)")
    print(f"⏱️  Temps total: {elapsed:.1f}s ({elapsed/total_count:.2f}s par cellule)")
    print(f"🚀 Speedup: ~{n_workers:.1f}x (estimation)")
    
    # Statistiques de délais
    all_delays = []
    for result in results.values():
        if result['success']:
            all_delays.extend(result['delays'].values())
    
    if all_delays:
        avg_delay = sum(all_delays) / len(all_delays)
        min_delay = min(all_delays)
        max_delay = max(all_delays)
        
        print(f"\n⏱️  Statistiques de délais:")
        print(f"   • Minimum: {min_delay:.2f} ps")
        print(f"   • Maximum: {max_delay:.2f} ps")
        print(f"   • Moyenne: {avg_delay:.2f} ps")

    # Export optionnel
    print(f"\n💾 Exporter les résultats? (o/n) [n]: ", end='')
    export = input().lower()
    if export in ['o', 'y', 'yes', 'oui']:
        output_file = Path("test_results.csv")
        with open(output_file, 'w') as f:
            f.write("Cell,Success,tphl_avg,tplh_avg,Error\n")
            for cell, result in results.items():
                if result['success']:
                    delays = result['delays']
                    tphl_values = [v for k, v in delays.items() if 'tphl' in k.lower()]
                    tplh_values = [v for k, v in delays.items() if 'tplh' in k.lower()]
                    tphl = sum(tphl_values) / len(tphl_values) if tphl_values else 0
                    tplh = sum(tplh_values) / len(tplh_values) if tplh_values else 0
                    f.write(f"{cell},True,{tphl:.3f},{tplh:.3f},\n")
                else:
                    error = result['error'].replace(',', ';')
                    f.write(f"{cell},False,,,{error}\n")
        print(f"✅ Résultats exportés vers: {output_file}")

    print("\n" + "="*80)
    print("✅ TESTS TERMINÉS")
    print("="*80)

if __name__ == "__main__":
    main()
