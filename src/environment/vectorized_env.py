# src/environment/vectorized_env.py
"""
Environnements vectorisés pour paralléliser l'entraînement RL
"""

from typing import Dict
from pathlib import Path
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from .gym_env import StandardCellEnv
from ..simulation.pdk_manager import PDKManager
from ..simulation.netlist_generator import SimulationConfig


def make_env(
    cell_name: str,
    pdk_name: str,  # ✅ Juste le nom du PDK
    config: SimulationConfig,
    cost_weights: Dict[str, float],
    max_steps: int,
    use_cache: bool,
    seed: int = None
):
    """
    Factory function pour créer un environnement
    
    Note: Chaque subprocess doit recréer son propre PDKManager
    """
    def _init():
        # ✅ Recréer PDKManager dans le subprocess (sans passer pdk_root)
        from ..simulation.pdk_manager import PDKManager
        
        # ✅ Utiliser la signature correcte : (pdk_name, use_uv, verbose)
        pdk = PDKManager(
            pdk_name=pdk_name,
            use_uv=True,
            verbose=False  # ✅ Pas de verbose dans les subprocesses
        )
        
        env = StandardCellEnv(
            cell_name=cell_name,
            pdk=pdk,
            config=config,
            cost_weights=cost_weights,
            max_steps=max_steps,
            verbose=False,
            use_cache=use_cache,
            
        )
        
        if seed is not None:
            env.reset(seed=seed)
        
        return env
    
    return _init


class VectorizedStandardCellEnv:
    """
    Wrapper pour créer N environnements en parallèle avec SubprocVecEnv
    """

    def __init__(
        self,
        cell_name: str,
        pdk: PDKManager,
        config: SimulationConfig,
        cost_weights: Dict[str, float] = None,
        max_steps: int = 50,
        n_envs: int = 4,
        use_cache: bool = True,
        use_subprocess: bool = True  # ✅ Option pour choisir SubprocVecEnv ou DummyVecEnv
    ):
        """
        Args:
            cell_name: Nom de la cellule à optimiser
            pdk: Instance PDKManager (utilisé seulement pour récupérer le nom)
            config: Configuration de simulation
            cost_weights: Poids pour la fonction de coût
            max_steps: Nombre max de steps par épisode
            n_envs: Nombre d'environnements parallèles
            use_cache: Utiliser le cache de simulations
            use_subprocess: Utiliser SubprocVecEnv (True) ou DummyVecEnv (False)
        """
        self.n_envs = n_envs
        self.cell_name = cell_name
        self.use_subprocess = use_subprocess
        
        if cost_weights is None:
            cost_weights = {'delay': 0.5, 'energy': 0.3, 'area': 0.2}
        
        # ✅ Créer les factory functions
        env_fns = [
            make_env(
                cell_name=cell_name,
                pdk_name=pdk.pdk_name,  # ✅ Passer juste le nom du PDK
                config=config,
                cost_weights=cost_weights,
                max_steps=max_steps,
                use_cache=use_cache,
                seed=42 + i
            )
            for i in range(n_envs)
        ]
        
        # ✅ Créer les envs vectorisés
        if use_subprocess and n_envs > 1:
            self.vec_env = SubprocVecEnv(env_fns, start_method='fork')
            print(f"🔀 Environnements vectorisés: {n_envs} envs en parallèle (SubprocVecEnv)")
        else:
            self.vec_env = DummyVecEnv(env_fns)
            print(f"🔀 Environnements vectorisés: {n_envs} envs séquentiels (DummyVecEnv)")

    def reset(self):
        """Reset tous les environnements"""
        return self.vec_env.reset()

    def step(self, actions):
        """Execute un step sur tous les environnements"""
        return self.vec_env.step(actions)

    def close(self):
        """Ferme tous les environnements"""
        self.vec_env.close()

    def __getattr__(self, name):
        """Délègue les attributs inconnus au vec_env"""
        return getattr(self.vec_env, name)


def create_vectorized_env(
    cell_name: str,
    pdk: PDKManager,
    config: SimulationConfig,
    n_envs: int = 4,
    **kwargs
) -> VectorizedStandardCellEnv:
    """
    Helper function pour créer rapidement un environnement vectorisé
    
    Example:
        >>> from src.simulation.pdk_manager import PDKManager
        >>> pdk = PDKManager("sky130")
        >>> vec_env = create_vectorized_env("inv_1", pdk, config, n_envs=4)
    """
    return VectorizedStandardCellEnv(
        cell_name=cell_name,
        pdk=pdk,
        config=config,
        n_envs=n_envs,
        **kwargs
    )
