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

        # === OBJECTIVE FUNCTION ===
        self.objective = ObjectiveFunction(
            cell_name=cell_name,
            config=self.config,
            pdk=pdk,
            verbose=verbose,
            use_cache=use_cache
        )
        
        # === METADATA POUR RL_AGENT ===
        self.cell_full_name = cell_name
        self.cell_category = self.objective.wm._get_category(cell_name)

        # === TRANSISTORS ===
        self.transistor_specs = self.objective.generator.extract_transistor_specs(cell_name)
        self.original_widths = {k: v["w"] for k, v in self.transistor_specs.items()}
        self.original_lengths = {k: v["l"] for k, v in self.transistor_specs.items()}
        self.transistor_names = sorted(self.transistor_specs.keys())
        self.n_transistors = len(self.transistor_names)

        if self.n_transistors == 0:
            raise ValueError(f"Aucun transistor trouvé pour {cell_name}")

        self.objective.original_lengths = self.original_lengths
        self.objective.original_widths = self.original_widths

        # === CONTRAINTES PHYSIQUES SKY130 (MÈTRES) ===
        self.min_width = 420.0 * 1e-9    
        self.max_width = 5000000.0 * 1e-9 

        # === METRICS ===
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

        # === TARGETS ===
        self.target_keys = list(self.metrics_keys)

        # === OBSERVATION SPACE ===
        obs_dim = self.n_transistors + 2 * len(self.metrics_keys)
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # === ACTION SPACE ===
        self.action_space = spaces.Box(
            low=-0.2, high=0.2, shape=(self.n_transistors,), dtype=np.float32
        )

        # === ÉTATS INTERNES ===
        self.current_widths: Optional[Dict[str, float]] = None
        self.current_metrics: Optional[Dict[str, float]] = None
        self.targets: Optional[Dict[str, float]] = None
        self.step_count = 0
        self.history = {"costs": [], "rewards": [], "widths": [], "metrics": []}

        if self.verbose: print(f"✅ Env créé: {cell_name} ({self.n_transistors} transistors)")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:

        if seed is not None:
            self._seed = seed
            super().reset(seed=seed)
            np.random.seed(seed)

        # Reset des largeurs
        self.current_widths = {name: float(w) for name, w in self.original_widths.items()}
        self.step_count = 0
        self.history = {"costs": [], "rewards": [], "widths": [], "metrics": []}

        # Targets
        if options is not None:
            self.targets = {}
            for key in self.target_keys:
                if key in options:
                     self.targets[key] = float(options[key])
                else:
                    # Valeur par défaut lâche
                    self.targets[key] = 1.0 
        else:
            self.targets = {
                "delay_rise": float(np.random.uniform(20e-12, 150e-12)),
                "delay_fall": float(np.random.uniform(20e-12, 150e-12)),
                "slew_in":    float(np.random.uniform(10e-12, 100e-12)),
                "slew_out_rise": float(np.random.uniform(10e-12, 100e-12)),
                "slew_out_fall": float(np.random.uniform(10e-12, 100e-12)),
                "power_dyn":  float(np.random.uniform(1e-6, 1e-4)),
                "power_leak": float(np.random.uniform(1e-10, 1e-8)),
                "area_um2":   float(np.random.uniform(0.3, 3.0)) 
            }

        # Simulation initiale
        self.current_metrics = self.objective.evaluate(
            self.current_widths, 
            cost_weights=self.cost_weights
        )

        return self._get_observation(), self._make_info(self.current_metrics)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        
        # 1. Action
        action_clean = np.array([float(x) for x in np.asarray(action, dtype=np.float64)])
        new_widths = self._apply_action(action_clean)


        formatted_w = {k: f"{v*1e9:.0f}nm" for k, v in new_widths.items()}
        print(f"🔄 Step {self.step_count} | Action: {action_clean} | New Widths: {formatted_w}")

        # 2. Simulation
        metrics = self.objective.evaluate(
            new_widths, 
            cost_weights=self.cost_weights,
            min_width_nm=self.min_width*1e9,  
            max_width_nm=self.max_width*1e9,    
        )
        
        # Gestion ÉCHEC SIMULATION
        if metrics is None or metrics.get('cost') == self.objective.penalty_cost:
            # Même en cas d'échec, on doit renvoyer info avec cost et widths
            # pour que le callback ne plante pas, mais avec un mauvais coût.
            
            reward = -10.0
            terminated = False
            truncated = (self.step_count >= self.max_steps)
            
            # Construction d'un info complet pour le callback
            fail_metrics = metrics if metrics else {"cost": 1000.0}
            info = self._make_info(fail_metrics, reward)
            info["error"] = "simulation_failed"
            
            # On garde les anciennes largeurs dans l'env, ou on accepte l'échec
            # Ici on retourne l'observation actuelle
            return self._get_observation(), float(reward), terminated, truncated, info

        # Simulation OK
        self.current_metrics = metrics
        self.current_widths = new_widths

        # 3. Reward
        reward, goal_reached = self._compute_reward()

        # 4. Fin ?
        terminated = bool(goal_reached)
        truncated = (self.step_count >= self.max_steps)

        # 5. Info
        info = self._make_info(self.current_metrics, reward, action_clean)
        info["goal_reached"] = goal_reached

        return self._get_observation(), float(reward), terminated, truncated, info

    def _apply_action(self, action: np.ndarray) -> Dict[str, float]:
        new_widths = {}
        for i, name in enumerate(self.transistor_names):
            delta = float(action[i]) if np.isfinite(action[i]) else 0.0
            current_w = float(self.current_widths[name])
            new_w = current_w * (1.0 + delta)
            new_w = float(np.clip(new_w, self.min_width, self.max_width))
            new_widths[name] = new_w
        return new_widths

    def _compute_reward(self) -> Tuple[float, bool]:
        errors = []
        for key in self.metrics_keys:
            if key in self.targets:
                v = float(self.current_metrics.get(key, 1e9)) 
                t = float(self.targets[key])
                
                # --- PROTECTION CONTRE INF ---
                if v == float('inf') or v > 1e9: 
                    # Si la mesure est infinie (échec partiel), on met une erreur fixe max
                    rel_err = 10.0 
                else:
                    rel_err = abs(v - t) / (abs(t) + 1e-12)
                    # Cap l'erreur relative pour éviter l'explosion du gradient
                    rel_err = min(rel_err, 10.0)
                
                errors.append(rel_err)
        
        if not errors: return 0.0, False
        
        reward = -sum(errors)
        goal_reached = all(e <= self.tolerance for e in errors)
        
        if goal_reached:
            reward += 10.0
            
        return float(reward), goal_reached

    def _make_info(self, metrics: Dict, reward=None, action=None) -> Dict:
        """Helper pour construire le dictionnaire info complet"""
        info = {
            "metrics": metrics,
            "targets": self.targets,
            "widths": self.current_widths, # Indispensable pour RLAgent
            "cost": metrics.get("cost", 1000.0) # Indispensable pour RLAgent
        }
        if reward is not None: info["reward"] = reward
        return info

    def _get_observation(self) -> np.ndarray:
        widths_norm = [self._normalize_width(self.current_widths[n]) for n in self.transistor_names]
        metrics_norm = []
        targets_norm = []
        
        for key in self.metrics_keys:
            metrics_norm.append(self._normalize_metric(key, self.current_metrics.get(key, 0.0)))
            targets_norm.append(self._normalize_metric(key, self.targets.get(key, 0.0)))

        return np.array(widths_norm + metrics_norm + targets_norm, dtype=np.float32)

    def _normalize_width(self, w: float) -> float:
        if self.max_width <= self.min_width: return 1.0
        norm = (w - self.min_width) / (self.max_width - self.min_width)
        return float(np.clip(norm, 0.0, 1.0) * 10.0)

    def _normalize_metric(self, key: str, value: float) -> float:
        ranges = {
            "delay_rise": (1e-12, 500e-12), 
            "delay_fall": (1e-12, 500e-12),
            "slew_in": (1e-12, 200e-12), 
            "slew_out_rise": (1e-12, 200e-12),
            "slew_out_fall": (1e-12, 200e-12),
            "power_dyn": (1e-9, 1e-3), 
            "power_leak": (1e-12, 1e-6),
            "area_um2": (0.1, 10.0)
        }
        vmin, vmax = ranges.get(key, (0.0, 1.0))
        if value == float('inf'): return 10.0 # Pénalité max normalisée
        norm = (value - vmin) / (vmax - vmin)
        return float(np.clip(norm, 0.0, 1.0) * 10.0)