# src/environment/parallel_env.py
"""
Environnement parallèle pour entraînement RL avec stable-baselines3
Utilise SubprocVecEnv pour vraie parallélisation multi-processus
"""

from typing import Callable, Optional, Dict, Any
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from old.netlist_generator import SimulationConfig
from src.environment.gym_env import StandardCellEnv


def make_cell_env(
    cell_name: str,
    pdk_name: str = "sky130",
    config: Optional[SimulationConfig] = None,
    cost_weights: Optional[Dict[str, float]] = None,
    max_steps: int = 50,
    verbose: bool = False,
    use_cache: bool = True,
    rank: int = 0  # Pour différencier les seeds
) -> Callable[[], gym.Env]:
    """
    Factory function pour créer un environnement
    Nécessaire pour SubprocVecEnv (doit être picklable)
    """
    def _init() -> gym.Env:
        # Chaque process initialise son propre PDK
        pdk = PDKManager(pdk_name, verbose=False)
        
        if config is None:
            sim_config = SimulationConfig(
                vdd=1.8,
                temp=27,
                corner='tt',
                cload=10e-15,
                trise=100e-12,
                tfall=100e-12
            )
        else:
            sim_config = config
        
        if cost_weights is None:
            weights = {'delay': 0.5, 'energy': 0.3, 'area': 0.2}
        else:
            weights = cost_weights
        
        env = StandardCellEnv(
            cell_name=cell_name,
            pdk=pdk,
            config=sim_config,
            cost_weights=weights,
            max_steps=max_steps,
            verbose=verbose,
            use_cache=use_cache
        )
        
        # Seed différent pour chaque environnement
        env.reset(seed=42 + rank)
        
        return env
    
    return _init


class ParallelCellEnvManager:
    """
    Gestionnaire d'environnements parallèles pour l'optimisation de cellules
    """
    
    def __init__(
        self,
        cell_name: str,
        n_envs: int = 4,
        pdk_name: str = "sky130",
        config: Optional[SimulationConfig] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        max_steps: int = 50,
        verbose: bool = False,
        use_cache: bool = True,
        use_subprocess: bool = True
    ):
        """
        Args:
            cell_name: Nom de la cellule à optimiser
            n_envs: Nombre d'environnements parallèles
            use_subprocess: Si True, utilise SubprocVecEnv (vraie parallélisation)
                           Si False, utilise DummyVecEnv (séquentiel, pour debug)
        """
        self.cell_name = cell_name
        self.n_envs = n_envs
        self.use_subprocess = use_subprocess
        
        print(f"🔧 Création de {n_envs} environnements parallèles...")
        print(f"   Mode: {'Subprocess (Multi-CPU)' if use_subprocess else 'Dummy (Single-CPU)'}")
        
        # Créer les fonctions factory pour chaque environnement
        env_fns = [
            make_cell_env(
                cell_name=cell_name,
                pdk_name=pdk_name,
                config=config,
                cost_weights=cost_weights,
                max_steps=max_steps,
                verbose=verbose,
                use_cache=use_cache,
                rank=i
            )
            for i in range(n_envs)
        ]
        
        # Créer l'environnement vectorisé
        if use_subprocess:
            # Vraie parallélisation multi-processus
            self.vec_env = SubprocVecEnv(env_fns, start_method='spawn')
        else:
            # Séquentiel (utile pour debug)
            self.vec_env = DummyVecEnv(env_fns)
    
    def get_env(self):
        """Retourne l'environnement vectorisé"""
        return self.vec_env
    
    def close(self):
        """Ferme tous les environnements"""
        self.vec_env.close()


# Version simplifiée avec make_vec_env de SB3
def create_parallel_env(
    cell_name: str,
    n_envs: int = 4,
    pdk_name: str = "sky130",
    config: Optional[SimulationConfig] = None,
    cost_weights: Optional[Dict[str, float]] = None,
    max_steps: int = 50,
    use_subprocess: bool = True,
    seed: int = 42
):
    """
    Version simplifiée utilisant make_vec_env de stable-baselines3
    """
    env_id = "StandardCell-v0"  # ID fictif
    
    # On doit d'abord enregistrer l'environnement
    # Ou utiliser directement la factory
    
    def make_env(rank):
        return make_cell_env(
            cell_name=cell_name,
            pdk_name=pdk_name,
            config=config,
            cost_weights=cost_weights,
            max_steps=max_steps,
            verbose=False,
            use_cache=True,
            rank=rank
        )
    
    vec_env = SubprocVecEnv(
        [lambda i=i: make_env(i)() for i in range(n_envs)],
        start_method='spawn'
    ) if use_subprocess else DummyVecEnv(
        [lambda i=i: make_env(i)() for i in range(n_envs)]
    )
    
    return vec_env
