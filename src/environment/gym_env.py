# src/environment/gym_env.py
"""
Environnement Gymnasium pour l'optimisation de standard cells
Compatible avec multiprocessing et stable-baselines3

Auteurs : Vincent Cauquil (vincent.cauquil@cpe.fr)
          Léonard Anselme (leonard.anselme@cpe.fr)

Assisté par IA (Copilote - Claude 3.5 - Gemini Pro)

Date : Novembre 2025 - Janvier 2026

class StandardCellEnv(gym.Env):
    __init__ : Initialisation de l'env
    reset : Reset de l'env
    step : Exécution d'une étape de l'env avec action donnée 
    _apply_action : Applique l'action aux largeurs des transistors
    _compute_reward_V1_2 : Calcul du reward (version 1.2)
    _compute_reward_V1_1 : Calcul du reward (version 1.1, obsolète)
    _make_info : Construit le dictionnaire info 
    _get_observation : Construit l'observation normalisée
    _normalize_width : Normalise une largeur de transistor
    _normalize_metric : Normalise une métrique spécifique
"""

# Importations nécessaires 
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional, Tuple, Any

# Ajout du chemin racine pour les imports locaux
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imports locaux
from src.optimization.objective import ObjectiveFunction
from src.simulation.netlist_generator import SimulationConfig
from src.simulation.pdk_manager import PDKManager

class StandardCellEnv(gym.Env):
    """
    Environnement Gym pour optimisation de standard cells via RL.
    Experimentation principal sur sky130_fd_sc_hd. (inv, nand, nor, etc. ~ 119 cellules)

    Args:
        cell_name (str): Nom de la cellule standard à optimiser
        pdk (PDKManager): Instance du gestionnaire PDK
        config (SimulationConfig, optional): Configuration de simulation NGSpice
        cost_weights (Dict[str, float], optional): Pondérations pour la fonction de coût
        max_steps (int): Nombre maximum d'étapes par épisode
        tolerance (float): Tolérance relative pour considérer la cible atteinte
        use_cache (bool): Utiliser le cache de simulations pour accélérer
        verbose (bool): Afficher les logs détaillés
        seed (int, optional): Graine aléatoire pour reproductibilité
        penality_rw (float): Récompense en cas d'échec de simulation
        mode (str): Mode d'utilisation ("training" ou "inference")

    Returns:
        gym.Env: Environnement Gymnasium pour l'optimisation de standard cells
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        cell_name: str,
        pdk: PDKManager,
        config: Optional[SimulationConfig] = None,
        cost_weights: Optional[Dict[str, float]] = None,
        target_ranges: Optional[Dict[str, Tuple[float, float]]] = None, # <--- NOUVEAU
        max_steps: int = 50,
        tolerance: float = 0.10,
        use_cache: bool = True,
        verbose: bool = False,
        seed: Optional[int] = None,
        penality_rw: float = -10.0,
        mode: str = "training",
    ):
        super().__init__()

        # === CONFIGURATION GÉNÉRALE ===
        self.cell_name = cell_name                  # Nom de la cellule standard à optimiser
        self.pdk = pdk                              # Gestionnaire PDK 
        self.config = config or SimulationConfig()  # Config de simulation NGSpice 
        self.cost_weights = cost_weights            # Pondérations pour la fonction de coût
        self.max_steps = max_steps                  # Nombre max d'étapes par épisode
        self.verbose = verbose                      # Affichage des logs détaillés
        self.tolerance = tolerance                  # Tolérance relative pour considérer la cible atteinte
        self.penality_rw = penality_rw              # Récompense en cas d'échec de simulation
        self.pdk_name = self.pdk.pdk_name           # Nom du PDK utilisé
        self.mode = mode                            # Mode d'utilisation ("training" ou "inference")

        # === RANDOM SEED ===
        # Gestion de la graine pour multiprocessing
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
        self.generation_ranges = {
            "delay_rise": (20e-12, 150e-12),
            "delay_fall": (20e-12, 150e-12),
            "slew_in":    (10e-12, 100e-12),
            "slew_out_rise": (10e-12, 100e-12),
            "slew_out_fall": (10e-12, 100e-12),
            "power_dyn":  (1e-6, 1e-4),
            "power_leak": (1e-10, 1e-8),
            "area_um2":   (0.3, 3.0) 
        }
        # Si le worker nous donne des ranges, on écrase les défauts
        if target_ranges:
            self.generation_ranges.update(target_ranges)


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

        # Séparation NMOS / PMOS
        self.nmos_names = []
        self.pmos_names = []
        for name, spec in self.transistor_specs.items():
            model_type = spec.get('type', '').lower()
            if 'nfet' in model_type:
                self.nmos_names.append(name)
            elif 'pfet' in model_type:
                self.pmos_names.append(name)

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

        if self.verbose: 
            print(f"✅ Env créé: {cell_name} ({self.n_transistors} transistors)")
            print(f"🔍 Détecté : {len(self.nmos_names)} NMOS, {len(self.pmos_names)} PMOS")

    def reset( 
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """ 
        Reset de l'environnement pour un nouvel épisode.

        Args:
            seed (int, optional): Graine aléatoire pour reproductibilité
            options (Dict, optional): Options supplémentaires pour le reset
        Returns:
            tuple: observation (np.ndarray), info (Dict)
        """

        if seed is not None:
            self._seed = seed
            super().reset(seed=seed)
            np.random.seed(seed)

        # Reset des largeurs
        self.current_widths = {name: float(w) for name, w in self.original_widths.items()}
        self.step_count = 0
        self.history = {"costs": [], "rewards": [], "widths": [], "metrics": []}
        self.targets = {}

        # Targets
        if self.mode == "inference":
            if self.verbose : print(f"🎯 Mode Inférence activé. Cibles reçues : {options}")
            # En mode inférence, on DOIT recevoir des options, ou utiliser une baseline
            if options:
                for key in self.target_keys:
                    if key in options:
                        self.targets[key] = float(options[key])
                    else:
                        self.targets[key] = 1.0
            else:
                print("⚠️ Attention: Mode inférence sans options fournies !")
                
        elif self.mode == "training":
            # Mode "training" (Défaut)
            if self.verbose : print("🎯 Mode Training activé. Génération de cibles aléatoires.")
            for key in self.target_keys:
                vmin, vmax = self.generation_ranges.get(key, (0.0, 1.0))
                if key == "power_leak": 
                    self.targets[key] = float(np.random.uniform(1e-10, 1e-8))
                else:
                    self.targets[key] = float(np.random.uniform(vmin, vmax))

        # Simulation initiale
        self.current_metrics = self.objective.evaluate(
            self.current_widths, 
            cost_weights=self.cost_weights
        )

        return self._get_observation(), self._make_info(self.current_metrics)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Exécute une étape de l'environnement avec l'action donnée.
        
        Args:
            action (np.ndarray): Tableau des actions pour chaque transistor
        Returns:    
            tuple: observation (np.ndarray), reward (float), terminated (bool), truncated (bool), info (Dict)
        """

        self.step_count += 1
        
        # 1. Action
        action_clean = np.array([float(x) for x in np.asarray(action, dtype=np.float64)])
        new_widths = self._apply_action(action_clean)

        formatted_w = {k: f"{v*1e9:.0f}nm" for k, v in new_widths.items()}
        if self.verbose : 
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
            
            reward = self.penality_rw # Pénalité fixe pour échec de simulation
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
        reward, goal_reached = self._compute_reward_V1_2()

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

    def _compute_reward_V1_2(self) -> Tuple[float, bool]:
        """
        Fonction de récompense intelligente avec contraintes physiques et pondération quadratique.
        Version 1.2
        - Pénalités quadratiques sur les erreurs relatives
        - Pondération plus forte pour les cibles strictes (faible delay/power)
        - Pénalité si ratio P/N invalide (PMOS < NMOS)
        - Bonus de succès augmenté

        Retourne : reward (float), is_success (bool)
        """
        reward = 0.0
        errors = []
        
        # 1. Calcul des erreurs pondérées sur les métriques
        for key in self.metrics_keys:
            if key in self.targets:
                # Valeur actuelle et Cible
                val = float(self.current_metrics.get(key, 1e9))
                tgt = float(self.targets[key])
                
                # Protection contre l'infini (simulation crashée)
                if val == float('inf') or val > 1e9:
                    rel_err = 5.0 # Punition fixe pour permettre le gradient
                else:
                    # Erreur relative : |Val - Cible| / Cible
                    rel_err = abs(val - tgt) / (abs(tgt) + 1e-12)
                
                errors.append(rel_err)
                
                # --- PONDÉRATION INTELLIGENTE ---
                weight = 1.0
                
                if "delay" in key and tgt < 50e-12:
                    weight = 5.0
                elif "power" in key and tgt < 1e-12:
                    weight = 5.0
                
                # --- PÉNALITÉ QUADRATIQUE ---
                # On met l'erreur au carré. 
                # Exemple : Erreur 0.1 -> 0.01. Erreur 1.0 -> 1.0 (grave).
                # On cap l'erreur à 2.0 avant le carré pour éviter l'explosion
                capped_err = min(rel_err, 2.0)
                reward -= weight * (capped_err ** 2)

        # --- PÉNALITÉ PHYSIQUE (Ratio P/N) ---
        # Un inverseur doit avoir un PMOS plus large ou égal au NMOS
        if self.current_widths:
            # Calcul des largeurs moyennes (en mètres)
            avg_wn,avg_wp = 0.0, 0.0
            
            if self.nmos_names:
                avg_wn = float(np.mean([self.current_widths[n] for n in self.nmos_names]))
            
            if self.pmos_names:
                avg_wp = float(np.mean([self.current_widths[n] for n in self.pmos_names]))
            
            # Application de la pénalité si les PMOS sont en moyenne plus petits que les NMOS
            if avg_wn > 0 and avg_wp > 0:
                if avg_wp < avg_wn:
                    # Pénalité proportionnelle à l'écart relatif vis à vis du ratio P/N
                    ratio_penalty = (avg_wn - avg_wp) / avg_wn 
                    reward -= 5.0 * ratio_penalty

        # --- BONUS DE SUCCÈS BOOSTÉ ---
        is_success = all(e <= self.tolerance for e in errors)
        if is_success:
            reward += 20.0  # On double la récompense (était 10.0)
            
        return float(reward), is_success

    # def _compute_reward_V1_1(self) -> Tuple[float, bool]:
    #     """
    #     Ancienne fonction de récompense simple 
    #     Version 1.1
    #     - Pénalités linéaires sur les erreurs relatives
    #     - Bonus de succès simple

    #     Retourne : reward (float), is_success (bool)
    #     """
    #     errors = []
    #     for key in self.metrics_keys:
    #         if key in self.targets:
    #             v = float(self.current_metrics.get(key, 1e9)) 
    #             t = float(self.targets[key])
                
    #             # --- PROTECTION CONTRE INF ---
    #             if v == float('inf') or v > 1e9: 
    #                 # Si la mesure est infinie (échec partiel), on met une erreur fixe max
    #                 rel_err = 10.0 
    #             else:
    #                 rel_err = abs(v - t) / (abs(t) + 1e-12)
    #                 # Cap l'erreur relative pour éviter l'explosion du gradient
    #                 rel_err = min(rel_err, 10.0)
                
    #             errors.append(rel_err)
        
    #     if not errors: return 0.0, False
        
    #     reward = -sum(errors)
    #     goal_reached = all(e <= self.tolerance for e in errors)
        
    #     if goal_reached:
    #         reward += 10.0
            
    #     return float(reward), goal_reached

    def _make_info(self, metrics: Dict, reward=None, action=None) -> Dict:
        """
        Helper pour construire le dictionnaire info complet
        
        returns : Dict
        """

        info = {
            "metrics": metrics,
            "targets": self.targets,
            "widths": self.current_widths, 
            "cost": metrics.get("cost", 1000.0) 
        }
        if action is not None: info["action"] = action
        if reward is not None: info["reward"] = reward

        return info

    def _get_observation(self) -> np.ndarray:
        """
        Construit l'observation normalisée pour l'agent RL
        
        returns : np.ndarray de shape (obs_dim,)
        """

        widths_norm = [self._normalize_width(self.current_widths[n]) for n in self.transistor_names]
        metrics_norm = []
        targets_norm = []
        
        for key in self.metrics_keys:
            metrics_norm.append(self._normalize_metric(key, self.current_metrics.get(key, 0.0)))
            targets_norm.append(self._normalize_metric(key, self.targets.get(key, 0.0)))

        return np.array(widths_norm + metrics_norm + targets_norm, dtype=np.float32)

    def _normalize_width(self, w: float) -> float:
        """
        Normalise la largeur d'un transistor entre 0.0 et 10.0
        
        returns : float 
        """
        if self.max_width <= self.min_width: return 1.0
        norm = (w - self.min_width) / (self.max_width - self.min_width)
        return float(np.clip(norm, 0.0, 1.0) * 10.0)

    def _normalize_metric(self, key: str, value: float) -> float:
        """
        Normalise une métrique entre 0.0 et 10.0 selon des bornes réalistes
        pour chaque type de métrique.

        returns : float
        """
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
        if value == float('inf'): 
            return 10.0 
        norm = (value - vmin) / (vmax - vmin)
        return float(np.clip(norm, 0.0, 1.0) * 10.0)