# src/optimization/simulation_cache.py
"""
Cache LRU pour les résultats de simulation
"""

from functools import lru_cache
import hashlib
import json
from typing import Dict, Tuple

class SimulationCache:
    """Cache des résultats de simulation avec clé = hash des widths"""
    
    def __init__(self, maxsize: int = 1000):
        self.cache = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0
    
    def _hash_widths(self, widths: Dict[str, float]) -> str:
        """Génère un hash unique pour une configuration"""
        # Arrondir à 1nm de précision pour éviter des variations infimes
        rounded = {k: round(v, 1) for k, v in sorted(widths.items())}
        key = json.dumps(rounded, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, widths: Dict[str, float]) -> Tuple[bool, Dict]:
        """
        Récupère un résultat du cache
        
        Returns:
            (found, measures) - found=True si dans le cache
        """
        key = self._hash_widths(widths)
        
        if key in self.cache:
            self.hits += 1
            return True, self.cache[key]
        else:
            self.misses += 1
            return False, {}
    
    def put(self, widths: Dict[str, float], measures: Dict[str, float]):
        """Stocke un résultat dans le cache"""
        key = self._hash_widths(widths)
        
        # LRU simple: supprimer le plus ancien si plein
        if len(self.cache) >= self.maxsize:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = measures.copy()
    
    def stats(self) -> Dict:
        """Statistiques du cache"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self.cache)
        }
    
    def clear(self):
        """Vide le cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
