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
    
    # Mapping type de cellule -> catégorie
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
    
    def __init__(self, base_dir: Path = None):
        """
        Args:
            base_dir: Répertoire racine (défaut: src/models/training_weights)
        """
        if base_dir is None:
            base_dir = Path(__file__).parent / "training_weights"
        
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer l'index s'il n'existe pas
        self.index_file = self.base_dir / "index.json"
        if not self.index_file.exists():
            self._init_index()
    
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
        """
        Extrait le type de cellule depuis son nom

            sky130_fd_sc_hd__xor2_1 -> xor2_1
        """
        # Enlever le préfixe PDK
        return cell_name.split('__')[-1]  # xor2_1
         
    
    def _get_category(self, cell_name: str) -> str:
        """Détermine la catégorie d'une cellule"""
        cell_type = self._get_cell_type(cell_name)
        
        # Chercher dans le mapping
        for key, category in self.CELL_CATEGORIES.items():
            if cell_type.startswith(key):
                return category
        
        # Catégorie par défaut
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
        """
        Sauvegarde les poids optimisés
        
        Args:
            cell_name: Nom complet de la cellule
            widths: Liste des multiplieurs de largeur
            metrics: Métriques de performance
            training_info: Infos d'entraînement (optionnel)
            algorithm: Algorithme utilisé
            episode: Numéro d'épisode (optionnel)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        # Déterminer la catégorie
        category = self._get_category(cell_name)
        cell_type = self._get_cell_type(cell_name)
        
        # Créer le répertoire de catégorie
        category_dir = self.base_dir / category
        category_dir.mkdir(exist_ok=True)
        
        # Construire le JSON
        data = {
            "cell_info": {
                "full_name": cell_name,
                "type": cell_type,
                "category": category,
                "n_transistors": len(widths)
            },
            "optimized_widths": {name: float(width) for name, width in widths.items()},
            "metrics": {
                "delay_avg_ps": float(metrics.get('delay_avg', 0)),
                "delay_tplh_ps": float(metrics.get('tplh_avg', 0)),
                "delay_tphl_ps": float(metrics.get('tphl_avg', 0)),
                "power_avg_uw": float(metrics.get('power_avg', 0)),
                "energy_dyn_fJ": float(metrics.get('energy_dyn', 0) * 1e15) if 'energy_dyn' in metrics else 0,
                "area_relative": float(metrics.get('area', 1.0))
            },
            "training": {
                "algorithm": algorithm,
                "timestamp": datetime.now().isoformat(),
                "cost_weights": training_info.get('cost_weights', {}),
                "config": {
                    "total_steps": training_info.get('total_steps', 0),
                    "best_cost": training_info.get('best_cost', 0.0),
                    "learning_rate": training_info.get('learning_rate', 3e-4),
                    "batch_size": training_info.get('batch_size', 64),
                    "n_epochs": training_info.get('n_epochs', 10),
                    "gamma": training_info.get('gamma', 0.99),
                    "gae_lambda": training_info.get('gae_lambda', 0.95),
                    "clip_range": training_info.get('clip_range', 0.2),
                    "ent_coef": training_info.get('ent_coef', 0.0),
                    "vf_coef": training_info.get('vf_coef', 0.5),
                    "max_grad_norm": training_info.get('max_grad_norm', 0.5),
                    "n_envs": training_info.get('n_envs', 1),
                    "training_time_seconds": training_info.get('training_time_seconds', 0.0),
                    "convergence": training_info.get('convergence', 'ongoing')
                }
            },
            "reference_metrics": metrics.get('reference', {}),
            "episode": episode,
            "metadata": metadata or {}

        }
        
        # Sauvegarder le fichier de poids
        filename = f"{cell_name}.json"
        filepath = category_dir / filename
        self._save_json(filepath, data)
        
        # Mettre à jour le metadata de la catégorie
        self._update_category_metadata(category, cell_type, data)
        
        # Mettre à jour l'index global
        self._update_global_index(category, cell_name)
        
        print(f"✅ Poids sauvegardés: {filepath}")
        return filepath
    
    def load_weights(self, json_path: Path) -> Dict[str, float]:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Avant : data["optimized_widths"] était une liste [899.75, 2529.66]
        # Maintenant : c'est déjà un Dict {"X0": 899.75, "X1": 2529.66}
        return data["optimized_widths"]

    
    def list_available_cells(self, category: str = None) -> List[str]:
        """
        Liste les cellules disponibles
        
        Args:
            category: Filtrer par catégorie (optionnel)
            
        Returns:
            Liste des noms de cellules
        """
        cells = []
        
        if category:
            categories = [category]
        else:
            categories = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        
        for cat in categories:
            cat_dir = self.base_dir / cat
            if not cat_dir.exists():
                continue
            
            for json_file in cat_dir.glob("*.json"):
                if json_file.name != "metadata.json":
                    data = self._load_json(json_file)
                    cells.append(data['cell_info']['full_name'])
        
        return sorted(cells)
    
    def get_category_summary(self, category: str) -> Dict:
        """Résumé des performances d'une catégorie"""
        metadata_file = self.base_dir / category / "metadata.json"
        
        if not metadata_file.exists():
            return {}
        
        return self._load_json(metadata_file)
    
    def _update_category_metadata(self, category: str, cell_type: str, data: Dict):
        """Met à jour le fichier metadata d'une catégorie"""
        category_dir = self.base_dir / category
        metadata_file = category_dir / "metadata.json"
        
        # Charger ou créer le metadata
        if metadata_file.exists():
            metadata = self._load_json(metadata_file)
        else:
            metadata = {
                "category": category,
                "cells": {},
                "statistics": {}
            }
        
        # Ajouter/mettre à jour la cellule
        metadata["cells"][cell_type] = {
            "full_name": data['cell_info']['full_name'],
            "last_updated": data['training']['timestamp'],
            "metrics": data['metrics'],
            "n_transistors": data['cell_info']['n_transistors']
        }
        
        # Calculer les statistiques
        all_delays = [c['metrics']['delay_avg_ps'] for c in metadata['cells'].values()]
        all_powers = [c['metrics']['power_avg_uw'] for c in metadata['cells'].values()]
        
        metadata["statistics"] = {
            "n_cells": len(metadata['cells']),
            "avg_delay_ps": float(np.mean(all_delays)),
            "avg_power_uw": float(np.mean(all_powers)),
            "last_updated": datetime.now().isoformat()
        }
        
        self._save_json(metadata_file, metadata)
    
    def _update_global_index(self, category: str, cell_name: str):
        """Met à jour l'index global"""
        index = self._load_json(self.index_file)
        
        if category not in index['categories']:
            index['categories'][category] = []
        
        if cell_name not in index['categories'][category]:
            index['categories'][category].append(cell_name)
            index['total_cells'] = sum(len(cells) for cells in index['categories'].values())
        
        index['last_updated'] = datetime.now().isoformat()
        self._save_json(self.index_file, index)
    
    def _save_json(self, filepath: Path, data: Dict):
        """Sauvegarde un JSON avec formatage"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=False)
    
    def _load_json(self, filepath: Path) -> Dict:
        """Charge un JSON"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def export_summary(self, output_file: Path = None):
        """Exporte un résumé global en CSV"""
        import pandas as pd
        
        if output_file is None:
            output_file = self.base_dir / "summary.csv"
        
        rows = []
        for cell_name in self.list_available_cells():
            data = self.load_weights(cell_name)
            if data:
                rows.append({
                    'cell': data['cell_info']['full_name'],
                    'category': data['cell_info']['category'],
                    'type': data['cell_info']['type'],
                    'n_transistors': data['cell_info']['n_transistors'],
                    'delay_ps': data['metrics']['delay_avg_ps'],
                    'power_uw': data['metrics']['power_avg_uw'],
                    'energy_fJ': data['metrics']['energy_dyn_fJ'],
                    'area': data['metrics']['area_relative'],
                    'algorithm': data['training']['algorithm'],
                    'date': data['training']['timestamp']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"✅ Résumé exporté: {output_file}")
        return df
