# src/environment/gym_env.py
"""
Environnement Gymnasium pour l'optimisation de standard cells
Compatible avec multiprocessing et stable-baselines3
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.objective import ObjectiveFunction
from src.simulation.netlist_generator import SimulationConfig
from src.simulation.pdk_manager import PDKManager


class StandardCellEnv(gym.Env):
    """
    Environnement Gym pour optimisation de standard cells
    
    État (observation):
        - Largeurs normalisées des transistors [n_trans]
        - Métriques normalisées: delay, energy, area [3]
        
    Action (continue):
        - Deltas de largeurs: [-0.2, +0.2] pour chaque transistor
        
    Récompense:
        - Basée sur l'amélioration du coût multi-objectif
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        cell_name: str,
        pdk: PDKManager,
        config: Optional[SimulationConfig] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        max_steps: int = 50,
        use_cache: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__()

        self.cell_name = cell_name
        self.pdk = pdk
        self.config = config or SimulationConfig()
        self.cost_weights = cost_weights or {
            'delay': 0.5,
            'energy': 0.3,
            'area': 0.2
        }
        self.max_steps = max_steps
        self.verbose = verbose
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
            # Si vous avez un RNG Gymnasium
            self.np_random = np.random.RandomState(seed)

        # Fonction objectif
        self.objective = ObjectiveFunction(
            cell_name=cell_name,
            config=self.config,
            pdk=pdk,
            verbose=verbose,
            use_cache=use_cache
        )

        # Largeurs originales
        self.original_widths = self.objective.get_original_widths()
        self.transistor_names = sorted(self.original_widths.keys())
        self.n_transistors = len(self.transistor_names)

        if self.n_transistors == 0:
            raise ValueError(f"Aucun transistor trouvé pour {cell_name}")

        # Contraintes physiques (Sky130)
        self.min_width = 420.0   # nm (minimum DRC)
        self.max_width = 5000.0  # nm (raisonnable)

        # Espaces Gym
        # Observation: [widths_norm (n), delay_norm, energy_norm, area_norm]
        obs_dim = self.n_transistors + 3
        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action: deltas de largeurs [-0.2, +0.2] (±20%)
        self.action_space = spaces.Box(
            low=-0.2,
            high=0.2,
            shape=(self.n_transistors,),
            dtype=np.float32
        )

        # État interne
        self.current_widths: Optional[Dict[str, float]] = None
        self.reference_cost: Optional[float] = None
        self.reference_metrics: Optional[Dict[str, float]] = None
        self.step_count = 0
        self.best_cost = float('inf')
        self.best_widths: Optional[Dict[str, float]] = None
        
        # Historique
        self.history = {
            'costs': [],
            'rewards': [],
            'widths': [],
            'metrics': []
        }

        if self.verbose:
            print(f"✅ Env créé: {cell_name} ({self.n_transistors} transistors)")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Réinitialise l'environnement"""
        # Gérer le seed
        if seed is not None:
            self._seed = seed
            super().reset(seed=seed)
            np.random.seed(seed)

        # ✅ Partir des largeurs originales (dict Python avec float)
        self.current_widths = {
            k: float(v) for k, v in self.original_widths.items()
        }
        
        self.step_count = 0
        self.history = {
            'costs': [],
            'rewards': [],
            'widths': [],
            'metrics': []
        }

        # Évaluer l'état initial
        metrics_dict = self.objective.evaluate(
            self.current_widths,
            self.cost_weights
        )

        cost = float(metrics_dict.get('cost', float('inf')))
        metrics = self._clean_metrics(metrics_dict)


        observation = self._get_observation(metrics)
        info = self._make_info(cost, metrics)

        if self.verbose:
            print(f"🔄 Reset: cost={cost:.4f}")

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Applique une action (deltas de largeurs)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1

        # ✅ 1. Convertir action en float Python
        action = np.asarray(action, dtype=np.float64)
        action_clean = np.array([float(x) for x in action])

        # ✅ 2. Calculer nouvelles largeurs (dict avec float Python)
        new_widths = self._apply_action(action_clean)

        # ✅ 3. Évaluer
        metrics_dict = self.objective.evaluate(new_widths, self.cost_weights)
        cost = float(metrics_dict.get('cost', float('inf')))
        metrics = self._clean_metrics(metrics_dict)
        simulation_failed = (cost == float('inf'))

        # ✅ 4. Calculer récompense
        reward, terminated, truncated = self._compute_reward(
            cost,
            simulation_failed,
            self.best_cost
        )

        # ✅ 5. Mise à jour état
        if not simulation_failed:
            self.current_widths = new_widths

            # Meilleure solution trouvée
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_widths = new_widths.copy()

        # Historique
        self.history['costs'].append(cost)
        self.history['rewards'].append(float(reward))
        self.history['widths'].append(list(new_widths.values()))
        self.history['metrics'].append(metrics)

        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True

        observation = self._get_observation(metrics)
        info = self._make_info(cost, metrics, reward, action_clean)

        if self.verbose and self.step_count % 10 == 0:
            improvement = (1 - cost / self.reference_cost) * 100 if self.reference_cost else 0
            print(f"  Step {self.step_count}: cost={cost:.4f} ({improvement:+.1f}%), "
                  f"reward={reward:.2f}")

        return observation, float(reward), terminated, truncated, info

    def _apply_action(self, action: np.ndarray) -> Dict[str, float]:
        """
        Applique l'action aux largeurs actuelles
        
        Returns:
            Dict avec float Python natifs
        """
        new_widths = {}
        
        for i, name in enumerate(self.transistor_names):
            current_w = self.current_widths[name]
            delta_fraction = float(action[i])  # [-0.2, +0.2]

            # Appliquer le delta
            new_w = current_w * (1.0 + delta_fraction)

            # Clipper aux contraintes DRC
            new_w = float(np.clip(new_w, self.min_width, self.max_width))

            new_widths[name] = new_w

        return new_widths

    def _compute_reward(
        self,
        cost: float,
        simulation_failed: bool,
        previous_cost: float
    ) -> Tuple[float, bool, bool]:
        """
        Calcule la récompense basée sur l'amélioration du coût

        Returns:
            (reward, terminated, truncated)
        """
        # Échec de simulation
        if simulation_failed or cost == float('inf'):
            return -10.0, False, False

        # Récompense basée sur l'amélioration relative
        if previous_cost > 0 and previous_cost != float('inf'):
            improvement = (previous_cost - cost) / previous_cost
            reward = 10.0 * improvement  # Scaling
        else:
            reward = 0.0

        # Bonus si nouveau meilleur
        if cost < self.best_cost:
            reward += 5.0

        # Terminer si 30% d'amélioration par rapport à la référence
        if self.reference_cost is not None and self.reference_cost > 0:
            terminated = bool(cost < 0.7 * self.reference_cost)
        else:
            terminated = False
        
        truncated = False

        return float(reward), terminated, truncated

    def _get_observation(self, metrics: Dict[str, float]) -> np.ndarray:
        """
        Construit le vecteur d'observation
        
        Returns:
            np.ndarray de float32
        """
        # Largeurs normalisées
        widths_norm = [
            self.current_widths[name] / self.original_widths[name]
            for name in self.transistor_names
        ]

        # Métriques normalisées
        if self.reference_metrics:
            delay_ref = self.reference_metrics.get('delay', 1.0)
            energy_ref = self.reference_metrics.get('energy', 1.0)
            area_ref = self.reference_metrics.get('area', 1.0)
        else:
            delay_ref = energy_ref = area_ref = 1.0

        delay_norm = metrics.get('delay', delay_ref) / delay_ref
        energy_norm = metrics.get('energy', energy_ref) / energy_ref
        area_norm = metrics.get('area', area_ref) / area_ref

        obs = np.array(
            widths_norm + [delay_norm, energy_norm, area_norm],
            dtype=np.float32
        )

        return obs

    def _make_info(
        self,
        cost: float,
        metrics: Dict[str, float],
        reward: Optional[float] = None,
        action: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Crée le dictionnaire info avec types Python natifs
        """
        info = {
            'cost': float(cost),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'widths': {k: float(v) for k, v in self.current_widths.items()},
            'step': int(self.step_count),
            'best_cost': float(self.best_cost),
        }

        if reward is not None:
            info['reward'] = float(reward)

        if action is not None:
            info['action'] = action.tolist()

        if self.reference_cost is not None:
            improvement = (1 - cost / self.reference_cost) * 100
            info['improvement_%'] = float(improvement)

        return info

    def _clean_metrics(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Convertit tous les types numpy en float Python
        """
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, np.generic)):
                clean[k] = float(v)
            elif isinstance(v, (int, float)):
                clean[k] = float(v)
            else:
                clean[k] = v
        return clean

    def _get_default_metrics(self) -> Dict[str, float]:
        """Métriques par défaut en cas d'échec"""
        return {
            'delay': 1e-9,
            'energy': 1e-12,
            'area': 100.0
        }

    def get_summary(self) -> Dict[str, Any]:
        """Résumé de l'épisode"""
        return {
            'n_steps': self.step_count,
            'best_cost': float(self.best_cost),
            'best_widths': {k: float(v) for k, v in (self.best_widths or {}).items()},
            'improvement_%': float((1 - self.best_cost / self.reference_cost) * 100)
                if self.reference_cost else 0.0,
            'total_reward': float(sum(self.history['rewards'])),
            'costs': [float(c) for c in self.history['costs']],
            'rewards': [float(r) for r in self.history['rewards']],
        }
