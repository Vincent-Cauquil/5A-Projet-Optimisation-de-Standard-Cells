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
        tolerance: float = 0.10,
        use_cache: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__()

        # === CONFIGURATION GÉNÉRALE ===
        self.cell_name = cell_name
        self.pdk = pdk
        self.config = config or SimulationConfig()
        self.cost_weights = cost_weights
        self.max_steps = max_steps
        self.verbose = verbose
        self.tolerance = tolerance

        # === RANDOM SEED ===
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.default_rng(seed)

        # === OBJECTIVE FUNCTION (NGSpice + parsing) ===
        self.objective = ObjectiveFunction(
            cell_name=cell_name,
            config=self.config,
            pdk=pdk,
            verbose=verbose,
            use_cache=use_cache
        )
        
        # === METADATA POUR RL_AGENT ===
        self.cell_full_name = cell_name
        self.cell_category = self.objective.wm._get_category(cell_name) #

        # === TRANSISTORS ===
        self.transistor_specs = self.objective.generator.extract_transistor_specs(cell_name)
        self.original_widths = {k: v["w"] for k, v in self.transistor_specs.items()}
        self.original_lengths = {k: v["l"] for k, v in self.transistor_specs.items()}
        self.transistor_names = sorted(self.transistor_specs.keys())
        self.n_transistors = len(self.transistor_names)

        if self.n_transistors == 0:
            raise ValueError(f"Aucun transistor trouvé pour {cell_name}")
        print(self.transistor_specs)

        self.objective.original_lengths = self.original_lengths
        self.objective.original_widths = self.original_widths

        # === CONTRAINTES PHYSIQUES SKY130 ===
        self.min_width = 420.0 * 1e-9    
        self.max_width = 1000000.0 * 1e-9 

        # === METRICS Supportées (via NGSpice) ===
        # Ce sont les métriques que la simulation fournit
        self.metrics_keys = [
            "delay_rise",
            "delay_fall",
            "slew_in",
            "slew_out_rise",
            "slew_out_fall",
            "power_dyn",
            "power_leak",
            "area_um2",
        ]

        # === TARGETS POUR RL GOAL-CONDITIONED ===
        # (mêmes clés que les métriques simulées)
        self.target_keys = list(self.metrics_keys)

        # === OBSERVATION SPACE ===
        # Observation = 
        #   widths_norm (n_transistors)
        # + metrics_norm (7)
        # + targets_norm (7)
        obs_dim = self.n_transistors + 2* len(self.metrics_keys)

        self.observation_space = spaces.Box(
            low=0.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # === ACTION SPACE ===
        # Action = deltas de largeur [-20%, +20%]
        self.action_space = spaces.Box(
            low=-0.2,
            high=0.2,
            shape=(self.n_transistors,),
            dtype=np.float32
        )

        # === ÉTATS INTERNES ===
        self.current_widths: Optional[Dict[str, float]] = None
        self.current_metrics: Optional[Dict[str, float]] = None
        self.targets: Optional[Dict[str, float]] = None

        self.step_count = 0

        # === HISTORIQUE ===
        self.history = {
            "costs": [],
            "rewards": [],
            "widths": [],
            "metrics": []
        }

        if self.verbose: print(f"✅ Env créé: {cell_name} ({self.n_transistors} transistors)")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:

        # === Seed ===
        if seed is not None:
            self._seed = seed
            super().reset(seed=seed)
            np.random.seed(seed)

        # === Reset des largeurs (valeurs originales) ===
        self.current_widths = {
            name: float(w) for name, w in self.original_widths.items()
        }

        # === Reset internes ===
        self.step_count = 0
        self.history = {
            "costs": [],
            "rewards": [],
            "widths": [],
            "metrics": []
        }

        # === Targets ===
        if options is not None:
            # Utilisateur impose les cibles
            self.targets = {}
            for key in self.target_keys:
                val = options.get(key, None)
                if val is None:
                    raise ValueError(f"Target '{key}' absente dans options")
                self.targets[key] = float(val)
        else:
            # Échantillonnage aléatoire pendant training
            self.targets = {
                "delay_rise": float(np.random.uniform(10e-12, 100e-12)),
                "delay_fall": float(np.random.uniform(10e-12, 100e-12)),
                "slew_rise":  float(np.random.uniform(5e-12, 80e-12)),
                "slew_fall":  float(np.random.uniform(5e-12, 80e-12)),
                "power_dyn":  float(np.random.uniform(1e-6, 2e-3)),
                "power_leak": float(np.random.uniform(1e-9, 1e-6)),
                "area_um2":   float(np.random.uniform(0.5e-6, 5e-6))
            }

        # === Simulation initiale NGSpice ===
        self.current_metrics = self.objective.evaluate(
            self.current_widths, 
            cost_weights=self.cost_weights 
        )

        # === Observation initiale ===
        observation = self._get_observation()

        # === Info ===
        info = {
            "metrics": self.current_metrics,
            "targets": self.targets
        }

        if self.verbose: print("🔄 Reset OK")

        return observation, info


    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        self.step_count += 1

        # === 1. Action → nouvelles largeurs ===
        action = np.asarray(action, dtype=np.float64)
        action_clean = np.array([float(x) for x in action])
        new_widths = self._apply_action(action_clean)

        # === 2. Simulation ===
        metrics = self.objective.evaluate(
            new_widths, 
            cost_weights=self.cost_weights
        )

        # Si la simulation renvoie None → échec
        simulation_failed = (metrics is None)
        if simulation_failed:
            reward = -10.0
            terminated = False
            truncated = (self.step_count >= self.max_steps)

            # Observation malgré tout (avec les anciennes métriques)
            obs = self._get_observation()
            info = {"error": "simulation_failed"}

            return obs, reward, terminated, truncated, info

        # Simulation OK → on met à jour
        self.current_metrics = metrics
        self.current_widths = new_widths

        # === 3. Reward computation ===
        reward, goal_reached = self._compute_reward()

        # === 4. Terminated ? ===
        terminated = bool(goal_reached)

        # === 5. Truncated ? ===
        truncated = (self.step_count >= self.max_steps)

        # === 6. Observation ===
        obs = self._get_observation()

        # === 7. Info ===
        info = {
            "metrics": self.current_metrics,
            "targets": self.targets,
            "goal_reached": goal_reached
        }

        # === 8. Logging verbose ===
        if self.verbose and self.step_count % 10 == 0:
            print(f"Step {self.step_count}: reward={reward:.2f}, goal={goal_reached}")

        return obs, float(reward), terminated, truncated, info


    def _apply_action(self, action: np.ndarray) -> Dict[str, float]:
        """
        Applique un vecteur d'actions (deltas relatifs) aux largeurs actuelles.
        Retourne un dict {transistor: nouvelle_largeur}.
        """

        if len(action) != self.n_transistors:
            raise ValueError(
                f"Action dimension {len(action)} incohérente avec "
                f"{self.n_transistors} transistors."
            )

        new_widths = {}

        for i, name in enumerate(self.transistor_names):

            delta_fraction = float(action[i])

            # Sécurité contre NaN / inf
            if not np.isfinite(delta_fraction):
                delta_fraction = 0.0

            current_w = float(self.current_widths[name])

            # Appliquer delta relatif
            new_w = current_w * (1.0 + delta_fraction)

            # Clipper DRC
            new_w = float(np.clip(new_w, self.min_width, self.max_width))

            new_widths[name] = new_w

        return new_widths


    def _compute_reward(self) -> Tuple[float, bool]:
        """
        Calcul de la récompense basée sur la distance aux targets
        pour un RL de type goal-conditioned.

        Returns:
            reward, goal_reached
        """

        errors = []

        # Pour chaque métrique avec un target associé
        for key in self.metrics_keys:
            if key not in self.targets:
                continue

            v = float(self.current_metrics[key])        # valeur mesurée
            t = float(self.targets[key])                # cible

            # erreur relative |v - t| / |t|
            rel_err = abs(v - t) / (abs(t) + 1e-12)
            errors.append(rel_err)

        if not errors:
            # Cas improbable : aucune métrique à optimiser
            return 0.0, False

        # Reward = négatif de la somme des erreurs
        reward = -sum(errors)

        # Objectif atteint si toutes les erreurs < tolérance
        goal_reached = all(e <= self.tolerance for e in errors)

        if goal_reached:
            reward += 10.0  # Bonus de réussite

        return float(reward), goal_reached


    def _get_observation(self) -> np.ndarray:
        """
        Construit le vecteur d'observation :
        - widths normalisées
        - metrics normalisées
        - targets normalisées
        """

        # === 1. Largeurs normalisées ===
        widths_norm = [
            self._normalize_width(self.current_widths[name])
            for name in self.transistor_names
        ]

        # === 2. Métriques normalisées ===
        metrics_norm = []
        for key in self.metrics_keys:
            if key in self.current_metrics:
                v = self.current_metrics[key]
                metrics_norm.append(self._normalize_metric(key, v))
            else:
                # placeholder si une métrique manque
                metrics_norm.append(0.0)

        # === 3. Targets normalisés ===
        targets_norm = []
        for key in self.metrics_keys:
            if key in self.targets:
                t = self.targets[key]
                targets_norm.append(self._normalize_metric(key, t))
            else:
                targets_norm.append(0.0)

        # === 4. Observation finale ===
        obs = np.array(
            widths_norm + metrics_norm + targets_norm,
            dtype=np.float32
        )

        return obs


    def _normalize_width(self, w: float) -> float:
        """
        Normalise la largeur dans l'intervalle [0, 10].
        Évite les divisions par zéro et clippe si nécessaire.
        """
        if self.max_width <= self.min_width:
            return 1.0  # fallback sécurisé

        norm = (w - self.min_width) / (self.max_width - self.min_width)
        norm = np.clip(norm, 0.0, 1.0)

        return float(norm * 10.0)

    def _normalize_metric(self, key: str, value: float) -> float:
        """
        Normalise une métrique dans [0, 10] en fonction de bornes plausibles
        adaptées aux circuits Sky130.
        """
        ranges = {
            "delay_rise":  (1e-12, 200e-12),
            "delay_fall":  (1e-12, 200e-12),
            "slew_rise":   (0.5e-12, 200e-12),
            "slew_fall":   (0.5e-12, 200e-12),
            "power_dyn":   (1e-6,  5e-3),
            "power_leak":  (1e-12, 1e-6),
            "area_um2":    (0.1, 10.0)
        }

        if key not in ranges:
            return 0.0

        vmin, vmax = ranges[key]

        if vmax <= vmin:
            return 5.0

        norm = (value - vmin) / (vmax - vmin)
        norm = np.clip(norm, 0.0, 1.0)
        return float(norm * 10.0)

    def _get_default_metrics(self) -> Dict[str, float]:
        """Fallback safe metrics."""
        return {
            "delay_rise":  100e-12,
            "delay_fall":  100e-12,
            "slew_rise":    50e-12,
            "slew_fall":    50e-12,
            "power_dyn":     1e-3,
            "power_leak":    1e-8,
            "area_mm2":      2e-6
        }

    def _make_info(
        self,
        metrics: Dict[str, float],
        reward: Optional[float] = None,
        action: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:

        info = {
            "metrics": {k: float(v) for k, v in metrics.items()},
            "widths": {k: float(v) for k, v in self.current_widths.items()},
            "step": int(self.step_count),
            "targets": self.targets
        }

        if reward is not None:
            info["reward"] = float(reward)

        if action is not None:
            info["action"] = action.tolist()

        return info

    def get_summary(self) -> Dict[str, Any]:
        """Résumé de l'épisode"""
        return {
            "n_steps": self.step_count,
            "total_reward": float(sum(self.history["rewards"])),
            "metrics": self.current_metrics,
            "final_widths": self.current_widths,
            "targets": self.targets
        }
