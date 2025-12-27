#!/usr/bin/env python3
import sys
import json
import time
import re
import tempfile
from pathlib import Path
from multiprocessing import Pool, Manager, cpu_count

# Ajout du root au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.spice_runner import SpiceRunner
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig
from src.models.weight_manager import WeightManager

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

def is_supported(cell_name: str) -> bool:
    """Reprend exactement votre logique de filtrage pour les 8 catégories"""
    c = cell_name.lower()
    # Votre logique de détection de votre script de test :
    return any([
        '__inv_' in c,
        '__buf_' in c or '__clkbuf_' in c, 
        '__nand' in c,
        '__nor' in c,
        '__and' in c,
        '__or' in c,
        '__xor' in c,
        '__xnor' in c
    ])

def baseline_worker(args):
    """Worker parallèle pour caractériser une cellule unique"""
    cell_name, config_dict, counter, lock, total = args
    
    try:
        pdk = PDKManager("sky130", verbose=False)
        generator = NetlistGenerator(pdk)
        runner = SpiceRunner(pdk.pdk_root, verbose=False)
        config = SimulationConfig(**config_dict)

        temp_dir = Path(tempfile.mkdtemp(prefix="baseline_"))
        output_path = temp_dir / f"{cell_name}_baseline.spice"
        
        # Génération via votre méthode modifiable
        netlist = generator.generate_characterization_netlist(
            cell_name=cell_name,
            output_path=str(output_path),
            config=config
        )

        # Simulation NGSpice
        result = runner.run_simulation(netlist_path=netlist, verbose=False)

        # Extraction des largeurs
        widths_info = generator.extract_transistor_specs(cell_name)
        original_widths = {k: float(v["w"]) for k, v in widths_info.items()}

        with lock:
            counter.value += 1
            short_name = cell_name.replace('sky130_fd_sc_hd__', '')
            print(f"\r🚀 [{counter.value}/{total}] Caractérisation: {short_name:<35}", end='', flush=True)

        if result['success'] and result['measures']:
            return (cell_name, {
                'success': True,
                'metrics': result['measures'],
                'widths': original_widths
            })
        
        err = result.get('errors', ['No measures'])[0] if result.get('errors') else "Unknown"
        return (cell_name, {'success': False, 'error': err})

    except Exception as e:
        return (cell_name, {'success': False, 'error': str(e)})

def main():
    print("="*80)
    print("🎯 GÉNÉRATION DES BASELINES (REFERENCE 119 CELLULES)")
    print("="*80)

    pdk = PDKManager("sky130")
    wm = WeightManager() 
    config = SimulationConfig(vdd=1.8, temp=27, corner="tt", cload=10e-15)
    
    config_dict = {
        'vdd': config.vdd, 'temp': config.temp, 'corner': config.corner,
        'cload': config.cload, 'trise': config.trise, 'tfall': config.tfall,
        'test_duration': config.test_duration, 'settling_time': config.settling_time
    }

    spice_lib = pdk.pdk_root / "libs.ref" / "sky130_fd_sc_hd" / "spice" / "sky130_fd_sc_hd.spice"
    all_cells = extract_all_cells(spice_lib)
    
    # Application du filtre strict (119 attendues)
    cells_to_run = [c for c in all_cells if is_supported(c)]
    total = len(cells_to_run)
    print(f"✅ {total} cellules identifiées pour la baseline.")

    n_workers = max(1, cpu_count() - 1)
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    task_args = [(cell, config_dict, counter, lock, total) for cell in cells_to_run]

    start_time = time.time()
    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(baseline_worker, task_args)
    
    # Structuration et Sauvegarde par catégorie
    reference_root = Path("src/models/references")
    reference_root.mkdir(parents=True, exist_ok=True)
    
    # Dictionnaire temporaire pour grouper les résultats
    grouped_results = {}
    success_count = 0
    for cell_name, res in raw_results:
        if res.get('success'):
            category = wm._get_category(cell_name) # Utilise le mapping officiel
            if category not in grouped_results:
                grouped_results[category] = {}
            
            grouped_results[category][cell_name] = {
                "metrics": res['metrics'],
                "widths": res['widths']
            }
            success_count += 1

    # Sauvegarde d'un fichier JSON par catégorie
    for category, cells in grouped_results.items():
        file_path = reference_root / f"{category}_baseline.json"
        with open(file_path, 'w') as f:
            json.dump(cells, f, indent=4)
        print(f"💾 Baseline catégorie '{category}' sauvegardée ({len(cells)} cellules)")
    print(f"\n\n{'='*80}")
    print(f"\n✅ TERMINÉ : {success_count}/{total} baselines générées dans {reference_root}")
    print(f"⏱️  Temps : {time.time() - start_time:.1f}s")
    print("="*80)

if __name__ == "__main__":
    main()