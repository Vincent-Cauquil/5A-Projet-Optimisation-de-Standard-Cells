# src/simulation/parallel_runner.py
"""
Runner parallèle pour accélérer l'entraînement RL
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile
import os
import multiprocessing as mp

from .spice_runner import SpiceRunner
from .pdk_manager import PDKManager


def _worker_simulate(args: Tuple[Path, Path, int]) -> Dict:
    """
    Fonction worker pour simulation parallèle
    
    Args:
        args: (netlist_path, pdk_root, worker_id)
        
    Returns:
        Dict avec success, measures, errors
    """
    netlist_path, pdk_root, worker_id = args
    
    # Créer un runner isolé pour ce worker
    runner = SpiceRunner(pdk_root, worker_id=worker_id)
    
    try:
        result = runner.run_simulation(netlist_path, verbose=False)
        return result
    except Exception as e:
        return {
            'success': False,
            'measures': {},
            'errors': [f"Worker {worker_id} exception: {str(e)}"]
        }


class ParallelSpiceRunner:
    """
    Exécute plusieurs simulations SPICE en parallèle
    """
    
    def __init__(self, pdk: PDKManager, n_workers: int = None):
        """
        Args:
            pdk: Manager PDK
            n_workers: Nombre de workers parallèles (défaut: CPU_COUNT - 1)
        """
        self.pdk = pdk
        
        if n_workers is None:
            cpu_count = mp.cpu_count()
            self.n_workers = max(1, cpu_count - 1)
        else:
            self.n_workers = max(1, n_workers)
        
        print(f"🚀 ParallelSpiceRunner: {self.n_workers} workers sur {mp.cpu_count()} CPUs")
    
    def run_batch(
        self, 
        netlists: List[Path], 
        verbose: bool = False
    ) -> List[Dict]:
        """
        Exécute un batch de netlists en parallèle
        
        Args:
            netlists: Liste de chemins de netlists
            verbose: Afficher la progression
            
        Returns:
            Liste de résultats {success, measures, errors}
        """
        if len(netlists) == 0:
            return []
        
        # Préparer les arguments pour chaque worker
        tasks = [
            (netlist, self.pdk.pdk_root, i) 
            for i, netlist in enumerate(netlists)
        ]
        
        results = [None] * len(netlists)
        completed = 0
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Soumettre toutes les tâches
            future_to_idx = {
                executor.submit(_worker_simulate, task): i
                for i, task in enumerate(tasks)
            }
            
            # Collecter les résultats au fur et à mesure
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                
                if verbose:
                    print(f"  [{completed}/{len(netlists)}] Simulations terminées", end='\r')
        
        if verbose:
            print()  # Nouvelle ligne après la barre de progression
        
        return results
    
    def run_single(self, netlist: Path) -> Dict:
        """
        Exécute une seule simulation (sans parallélisation)
        
        Args:
            netlist: Chemin de la netlist
            
        Returns:
            Résultat {success, measures, errors}
        """
        runner = SpiceRunner(self.pdk.pdk_root)
        return runner.run_simulation(netlist, verbose=False)


class BatchSimulator:
    """
    Simulateur batch pour entraînement RL avec parallélisation
    """
    
    def __init__(
        self, 
        pdk: PDKManager,
        n_workers: int = None,
        batch_size: int = 8
    ):
        """
        Args:
            pdk: Manager PDK
            n_workers: Nombre de workers parallèles
            batch_size: Taille des batchs pour parallélisation
        """
        self.parallel_runner = ParallelSpiceRunner(pdk, n_workers)
        self.batch_size = batch_size
        self.pending_netlists = []
        self.pending_callbacks = []
    
    def submit(self, netlist: Path, callback=None):
        """
        Ajoute une simulation au batch
        
        Args:
            netlist: Chemin de la netlist
            callback: Fonction appelée avec le résultat
        """
        self.pending_netlists.append(netlist)
        self.pending_callbacks.append(callback)
        
        # Si le batch est plein, exécuter
        if len(self.pending_netlists) >= self.batch_size:
            self.flush()
    
    def flush(self):
        """
        Exécute toutes les simulations en attente
        """
        if len(self.pending_netlists) == 0:
            return
        
        # Exécuter en parallèle
        results = self.parallel_runner.run_batch(
            self.pending_netlists,
            verbose=False
        )
        
        # Appeler les callbacks
        for result, callback in zip(results, self.pending_callbacks):
            if callback:
                callback(result)
        
        # Vider le batch
        self.pending_netlists = []
        self.pending_callbacks = []
    
    def __del__(self):
        """Exécuter les simulations restantes"""
        self.flush()
