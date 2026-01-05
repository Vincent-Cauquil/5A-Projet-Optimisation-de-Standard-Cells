#!/usr/bin/env python3
# src/optimization/simulation_cache.py
# ============================================================
#  Simulation Cache 
# ============================================================
"""
Cache thread-safe pour les simulations SPICE.
Compatible avec multiprocessing et types numpy pour éviter les simulations redondantes.

Auteurs : Vincent Cauquil (vincent.cauquil@cpe.fr)
          Léonard Anselme (leonard.anselme@cpe.fr)

Date : Novembre 2025 - Janvier 2026

class SimulationCache:
    get_cache_key : Génère une clé unique basée sur les largeurs et la config.
    get : Récupère une entrée du cache.
    set : Ajoute une nouvelle simulation au cache.
    stats : Fournit des statistiques sur le hit rate.
"""

import json
import hashlib
from typing import Dict, Optional, Any, List
import numpy as np


class SimulationCache:
    """
    Cache simple pour éviter de refaire les mêmes simulations
    Thread-safe et compatible multiprocessing
    """

    def __init__(self, max_size: int = 10000, precision: int = 3):
        """
        Args:
            max_size: Taille maximale du cache
            precision: Nombre de décimales après la virgule (en nanomètres)
                       3 décimales = 0.001 nm de précision (suffisant)
        """
        self.cache: Dict[str, Dict[str, float]] = {}
        self.max_size = max_size
        self.precision = precision
        self.hits = 0
        self.misses = 0

    def get_cache_key(
        self,
        widths: Dict[str, float],
        config: Any  # SimulationConfig
    ) -> str:
        """
        Génère une clé de cache unique.
        CORRIGÉ : Convertit en nanomètres avant d'arrondir pour éviter les collisions.
        """
        # 1. Nettoyage et conversion en nm
        # On multiplie par 1e9 pour passer en nm, puis on arrondit
        rounded_widths = {}
        for name, w in sorted(widths.items()):
            # Conversion sécurisée en float natif
            val = safe_float_conversion(w)
            # Conversion m -> nm et arrondi
            val_nm = round(val * 1e9, self.precision)
            rounded_widths[name] = val_nm
        
        # 2. Construction du dictionnaire de clé
        cache_data = {
            'widths': rounded_widths,
            'vdd': round(config.vdd, 3),
            'temp': round(config.temp, 1),
            'corner': str(config.corner),
            'cload': f"{config.cload:.2e}",
            'trise': f"{config.trise:.2e}",
            'tfall': f"{config.tfall:.2e}"
        }
        
        # 3. Hashage stable (tri des clés garanti par json.dumps)
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Dict[str, float]]:
        """Récupère une entrée du cache"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key].copy()
        else:
            self.misses += 1
            return None
        
    def set(self, key: str, metrics: Dict[str, float]) -> None:
        """Ajoute une entrée au cache"""
        if len(self.cache) >= self.max_size:
            # Nettoyage sommaire : supprimer 10% des entrées les plus anciennes
            # (Note: sur un dict python >3.7, l'ordre d'insertion est préservé)
            to_remove = int(self.max_size * 0.1)
            keys_to_del = list(self.cache.keys())[:to_remove]
            for k in keys_to_del:
                del self.cache[k]
        
        # On nettoie les métriques pour éviter les types numpy non sérialisables
        self.cache[key] = clean_dict_for_json(metrics)

    def stats(self) -> Dict[str, Any]:
        """Statistiques d'utilisation"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache),
            'max_size': self.max_size
        }

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"SimulationCache(size={stats['size']}/{stats['max_size']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )


# === FONCTIONS UTILITAIRES (VOUS LES AVIEZ, ELLES SONT UTILES !) ===

def safe_float_conversion(value: Any) -> float:
    """
    Convertit n'importe quel type numérique en float Python natif.
    Indispensable pour éviter les erreurs de sérialisation JSON avec Numpy.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        return float(value)
    elif isinstance(value, (int, float)):
        return float(value)
    elif hasattr(value, 'item'):  # Cas torch.Tensor si utilisé un jour
        return float(value.item())
    else:
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parcourt récursivement un dictionnaire pour convertir tous les types
    exotiques (numpy) en types natifs Python (float, list, dict).
    """
    clean = {}
    for k, v in data.items():
        if isinstance(v, dict):
            clean[k] = clean_dict_for_json(v)
        elif isinstance(v, (list, tuple)):
            clean[k] = [safe_float_conversion(x) if isinstance(x, (int, float, np.number)) else x for x in v]
        elif isinstance(v, (int, float, np.number)):
            clean[k] = safe_float_conversion(v)
        else:
            clean[k] = v
    return clean