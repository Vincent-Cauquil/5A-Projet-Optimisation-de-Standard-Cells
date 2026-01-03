# src/models/weight_manager.py
"""
Gestionnaire de sauvegarde/chargement des poids optimisés
Organise par catégorie de standard cell
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

class WeightManager:
    """Gère la sauvegarde et le chargement des poids optimisés"""
    
    CELL_CATEGORIES = {
        'inv': 'inv',
        'buf': 'buf',
        'nand2': 'nand',
        'nand3': 'nand',
        'nand4': 'nand',
        'nor2': 'nor',
        'nor3': 'nor',
        'nor4': 'nor',
        'and2': 'and',
        'and3': 'and',
        'and4': 'and',
        'or2': 'or',
        'or3': 'or',
        'or4': 'or',
        'xor2': 'xor',
        'xor3': 'xor',
        'xnor2': 'xnor',
        'xnor3': 'xnor',
        'mux2': 'mux',
        'mux4': 'mux',
        'dff': 'sequential',
        'dlatch': 'sequential',
    }
    
    def __init__(self, base_dir: Path = None, pdk_name: str = "sky130", config_data: Dict = None):
        """
        Args:
            base_dir: Répertoire racine (défaut: src/models/training_weights)
        """
        if base_dir is None:
            base_dir = Path(f"data/{pdk_name}/weight")

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer l'index s'il n'existe pas
        self.index_file = self.base_dir / "index.json"
        if not self.index_file.exists():
            self._init_index()
        self.config_data = config_data or {}
    
    def _init_index(self):
        """Initialise l'index global"""
        index = {
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "categories": {},
            "total_cells": 0
        }
        self._save_json(self.index_file, index)
    
    def _get_cell_type(self, cell_name: str) -> str:
        """Extrait le type de cellule depuis son nom"""
        return cell_name.split('__')[-1]
    
    def _get_category(self, cell_name: str) -> str:
        """Détermine la catégorie d'une cellule"""
        cell_type = self._get_cell_type(cell_name)
        for key, category in self.CELL_CATEGORIES.items():
            if cell_type.startswith(key):
                return category
        return "other"
    
    def save_weights(
        self,
        cell_name: str,
        widths: Dict[str, float],
        metrics: Dict[str, float],
        training_info: Optional[Dict] = None,
        algorithm: str = "PPO",
        metadata: Dict = None,
        episode: int = None
    ) -> Path:
        """Sauvegarde les poids optimisés"""
        category = self._get_category(cell_name)
        cell_type = self._get_cell_type(cell_name)
        
        category_dir = self.base_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # --- CORRECTION DES MAPPINGS ET UNITÉS ---
        # On convertit les unités pour correspondre aux suffixes (_ps, _fJ, etc.)
        # delay_avg (s) -> delay_avg_ps (ps) : * 1e12
        # energy_dyn (J) -> energy_dyn_fJ (fJ) : * 1e15

        data = {
            "cell_info": {
                "full_name": cell_name,
                "type": cell_type,
                "category": category,
                "n_transistors": len(widths)
            },
            "optimized_widths": {name: float(width) for name, width in widths.items()},
            "metrics": {
                "delay_avg_ps": float(metrics.get('delay_avg', 0) * 1e12),
                "delay_rise_ps": float(metrics.get('delay_rise', 0) * 1e12), 
                "delay_fall_ps": float(metrics.get('delay_fall', 0) * 1e12), 
                "energy_dyn_fJ": float(metrics.get('energy_dyn', 0) * 1e15),
                "power_leak_nW": float(metrics.get('power_leak', 0) * 1e9),
                "area_um2": float(metrics.get('area_um2', 0)),
                "cost": float(metrics.get('cost', 1.0))
            },
            "training": {
                "algorithm": algorithm,
                "timestamp": datetime.now().isoformat(),
                "cost_weights": training_info.get('cost_weights', {}),
                "training_steps": training_info.get('total_steps', 0),
                "best_cost": training_info.get('best_cost', 0.0),
                "training_time_seconds": training_info.get('training_time_seconds', 0.0),
                "config": {**self.config_data}
            },
            "episode": episode,
            "metadata": metadata or {}
        }
        
        filename = f"{cell_name}.json"
        filepath = category_dir / filename
        self._save_json(filepath, data)
        
        self._update_category_metadata(category, cell_type, data)
        self._update_global_index(category, cell_name)
        
        print(f"✅ Poids sauvegardés: {filepath}")
        return filepath
    
    def load_weights(self, json_path: Path) -> Dict[str, float]:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data["optimized_widths"]

    def list_available_cells(self, category: str = None) -> List[str]:
        cells = []
        if category:
            categories = [category]
        else:
            categories = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        
        for cat in categories:
            cat_dir = self.base_dir / cat
            if not cat_dir.exists(): continue
            for json_file in cat_dir.glob("*.json"):
                if json_file.name != "metadata.json":
                    data = self._load_json(json_file)
                    cells.append(data['cell_info']['full_name'])
        return sorted(cells)
    
    def get_category_summary(self, category: str) -> Dict:
        metadata_file = self.base_dir / category / "metadata.json"
        if not metadata_file.exists(): return {}
        return self._load_json(metadata_file)
    
    def _update_category_metadata(self, category: str, cell_type: str, data: Dict):
        category_dir = self.base_dir / category
        metadata_file = category_dir / "metadata.json"
        
        if metadata_file.exists():
            metadata = self._load_json(metadata_file)
        else:
            metadata = {"category": category, "cells": {}, "statistics": {}}
        
        metadata["cells"][cell_type] = {
            "full_name": data['cell_info']['full_name'],
            "last_updated": data['training']['timestamp'],
            "metrics": data['metrics'],
            "n_transistors": data['cell_info']['n_transistors']
        }
        
        # Stats simples
        all_delays = [c['metrics'].get('delay_avg_ps', 0) for c in metadata['cells'].values()]
        metadata["statistics"] = {
            "n_cells": len(metadata['cells']),
            "avg_delay_ps": float(np.mean(all_delays)) if all_delays else 0,
            "last_updated": datetime.now().isoformat()
        }
        
        self._save_json(metadata_file, metadata)
    
    def _update_global_index(self, category: str, cell_name: str):
        index = self._load_json(self.index_file)
        if category not in index['categories']:
            index['categories'][category] = []
        if cell_name not in index['categories'][category]:
            index['categories'][category].append(cell_name)
            index['total_cells'] = sum(len(cells) for cells in index['categories'].values())
        index['last_updated'] = datetime.now().isoformat()
        self._save_json(self.index_file, index)
    
    def _save_json(self, filepath: Path, data: Dict):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=False)
    
    def _load_json(self, filepath: Path) -> Dict:
        with open(filepath, 'r') as f:
            return json.load(f)