# src/optimization/simulation_cache.py
"""
Cache thread-safe pour les simulations SPICE
Compatible avec multiprocessing et numpy types
"""

import json
import hashlib
from typing import Dict, Optional, Any
import numpy as np


class SimulationCache:
    """
    Cache simple pour éviter de refaire les mêmes simulations
    Thread-safe et compatible multiprocessing
    """

    def __init__(self, max_size: int = 10000, precision: int = 9):
        """
        Args:
            precision: Nombre de décimales pour arrondir les largeurs
        """
        self.cache: Dict[str, Dict[str, float]] = {}
        self.max_size = max_size
        self.precision = precision
        self.hits = 0
        self.misses = 0

    def _hash_widths(self, widths: Dict[str, float]) -> str:
        """
        Crée une clé de hash unique pour un ensemble de largeurs
        Compatible avec numpy types
        """
        # ✅ Convertir tous les types numpy en float Python natif
        rounded = {
            k: float(round(float(v), self.precision))
            for k, v in widths.items()
        }
        
        # Trier les clés pour cohérence
        sorted_items = sorted(rounded.items())
        
        # Créer un hash
        key_string = json.dumps(sorted_items, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_cache_key(
        self,
        widths: Dict[str, float],
        config: Any  # SimulationConfig
    ) -> str:
        """
        Génère une clé de cache unique basée sur les paramètres.
        
        Args:
            widths: Largeurs des transistors
            config: Configuration de simulation
            
        Returns:
            Clé de cache (hash SHA256)
        """
        # Arrondir les largeurs à 0.1nm pour éviter les doublons
        rounded_widths = {
            name: round(w, 1) for name, w in sorted(widths.items())
        }
        
        # Créer un dict avec tous les paramètres pertinents
        cache_data = {
            'widths': rounded_widths,
            'vdd': round(config.vdd, 3),
            'temp': round(config.temp, 1),
            'corner': config.corner,
            'cload': f"{config.cload:.2e}",
            'trise': f"{config.trise:.2e}",
            'tfall': f"{config.tfall:.2e}"
        }
        
        # Générer hash
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Dict[str, float]]:
        """
        Récupère un résultat du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            Métriques si trouvées, None sinon
        """
        if key in self.cache:
            self.hits += 1
            return self.cache[key].copy()
        else:
            self.misses += 1
            return None
        
    def set(self, key: str, metrics: Dict[str, float]) -> None:
        """
        Ajoute un résultat au cache.
        
        Args:
            key: Clé de cache
            metrics: Métriques à stocker
        """
        # Vérifier la taille max
        if len(self.cache) >= self.max_size:
            # Supprimer 10% des entrées les plus anciennes
            to_remove = int(self.max_size * 0.1)
            for k in list(self.cache.keys())[:to_remove]:
                del self.cache[k]
        
        self.cache[key] = metrics.copy()

    def clear(self):
        """Vide le cache"""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.
        
        Returns:
            Dict avec hits, misses, taux de hit, taille
        """
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
        return len(self._cache)

    def __repr__(self) -> str:
        """Représentation textuelle"""
        stats = self.stats()
        return (
            f"SimulationCache(size={stats['size']}/{stats['max_size']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )

def safe_float_conversion(value: Any) -> float:
    """
    Convertit n'importe quel type numérique en float Python natif
    Compatible avec numpy, torch, etc.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        return float(value)
    elif isinstance(value, (int, float)):
        return float(value)
    elif hasattr(value, 'item'):  # torch.Tensor
        return float(value.item())
    else:
        return float(value)


def clean_dict_for_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nettoie un dictionnaire pour le rendre JSON-serializable
    """
    clean = {}
    for k, v in data.items():
        if isinstance(v, dict):
            clean[k] = clean_dict_for_json(v)
        elif isinstance(v, (list, tuple)):
            clean[k] = [safe_float_conversion(x) if isinstance(x, (np.ndarray, np.generic))
                       else x for x in v]
        elif isinstance(v, (np.ndarray, np.generic)):
            clean[k] = safe_float_conversion(v)
        else:
            clean[k] = v
    return clean
