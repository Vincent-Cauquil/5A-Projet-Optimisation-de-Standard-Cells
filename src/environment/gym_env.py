# # src/environment/gym_env.py
# """
# Environnement Gymnasium pour l'optimisation de standard cells
# Inspiré du TP RL (Q-Learning sur CliffWalking)
# """

# import gymnasium as gym
# import numpy as np
# from gymnasium import spaces
# from typing import Dict, Optional, Tuple, List
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))

# from src.optimization.objective import ObjectiveFunction
# from src.optimization.cell_modifier import CellModifier


# class CellOptimizationEnv(gym.Env):
#     """
#     Environnement RL pour optimiser les largeurs de transistors
    
#     **État (Observation)** :
#         - Métriques normalisées : [delay_norm, power_norm, area_norm]
#         - Largeurs actuelles normalisées : [w1_norm, w2_norm, ..., wN_norm]
    
#     **Action** :
#         - Vecteur continu de multiplieurs de largeur : [m1, m2, ..., mN]
#         - Chaque multiplier ∈ [0.5, 3.0]
    
#     **Récompense** :
#         - reward = -cost (pour maximiser on minimise le coût)
#         - Bonus si amélioration significative
#         - Pénalité si simulation échoue
    
#     **Terminaison** :
#         - Max steps atteint
#         - Objectif atteint (amélioration > seuil)
#         - Échec de simulation
#     """
    
#     metadata = {"render_modes": ["human"]}
    
#     def __init__(
#         self,
#         objective_func: ObjectiveFunction,
#         max_steps: int = 100,
#         success_threshold: float = 0.8,  # Coût < 80% de référence
#         action_scale: float = 0.2,       # Changement max par step
#         verbose: bool = False
#     ):
#         """
#         Args:
#             objective_func: Fonction objectif pour évaluer les cellules
#             max_steps: Nombre max d'itérations par épisode
#             success_threshold: Seuil de succès (coût relatif)
#             action_scale: Amplitude des changements (0.1 = ±10%)
#             verbose: Affichage debug
#         """
#         super().__init__()
        
#         self.objective = objective_func
#         self.modifier = objective_func.modifier
#         self.max_steps = max_steps
#         self.success_threshold = success_threshold
#         self.action_scale = action_scale
#         self.verbose = verbose
        
#         # Récupérer le nombre de transistors
#         self.n_transistors = self._get_n_transistors()
        
#         # Limites physiques des largeurs (en multiplieurs)
#         self.width_min = 0.5  # 50% de la largeur originale
#         self.width_max = 3.0  # 300% de la largeur originale
        
#         # ============ SPACES (comme dans Gymnasium) ============
        
#         # Action space : multiplieurs de largeur (continu)
#         # Chaque action ∈ [-1, 1], on la rescale ensuite
#         self.action_space = spaces.Box(
#             low=-1.0,
#             high=1.0,
#             shape=(self.n_transistors,),
#             dtype=np.float32
#         )
        
#         # Observation space : [métriques normalisées, largeurs normalisées]
#         # 3 métriques (delay, power, area) + N largeurs
#         obs_dim = 3 + self.n_transistors
#         self.observation_space = spaces.Box(
#             low=0.0,
#             high=3.0,  # Les métriques/largeurs peuvent aller jusqu'à 3x la ref
#             shape=(obs_dim,),
#             dtype=np.float32
#         )
        
#         # État interne
#         self.current_widths = None
#         self.reference_cost = None
#         self.reference_metrics = None
#         self.best_cost = float('inf')
#         self.best_widths = None
#         self.step_count = 0
#         self.episode_count = 0
        
#         # Historique (pour analyse)
#         self.history = {
#             'costs': [],
#             'rewards': [],
#             'actions': [],
#             'widths': []
#         }
    
#     def _get_n_transistors(self) -> int:
#         """Récupère le nombre de transistors de la cellule"""
#         try:
#             widths = self.modifier.get_current_widths()
#             return len(widths)
#         except Exception as e:
#             if self.verbose:
#                 print(f"⚠️  Impossible de récupérer le nombre de transistors: {e}")
#             return 4  # Valeur par défaut (ex: inverseur simple)
    
#     def reset(
#         self, 
#         seed: Optional[int] = None, 
#         options: Optional[dict] = None
#     ) -> Tuple[np.ndarray, Dict]:
#         """
#         Reset l'environnement (début d'épisode)
        
#         Returns:
#             observation: État initial
#             info: Infos supplémentaires
#         """
#         super().reset(seed=seed)
        
#         self.step_count = 0
#         self.episode_count += 1
        
#         # Réinitialiser à la largeur originale
#         try:
#             self.current_widths = np.array(self.modifier.get_original_widths())
            
#             # Évaluer le coût de référence (première fois seulement)
#             if self.reference_cost is None:
#                 ref_results = self.objective.evaluate(self.current_widths)
#                 self.reference_cost = ref_results['cost']
#                 self.reference_metrics = ref_results
                
#                 if self.verbose:
#                     print(f"\n📏 Référence établie:")
#                     print(f"   Cost: {self.reference_cost:.4f}")
#                     print(f"   Delay: {ref_results.get('delay_avg', 0):.2f} ps")
#                     print(f"   Power: {ref_results.get('power_avg', 0):.2f} µW")
            
#             # Réinitialiser le meilleur
#             self.best_cost = self.reference_cost
#             self.best_widths = self.current_widths.copy()
            
#         except Exception as e:
#             if self.verbose:
#                 print(f"❌ Erreur reset: {e}")
#             # Largeurs par défaut si échec
#             self.current_widths = np.ones(self.n_transistors) * 0.5
#             self.reference_cost = 1.0
#             self.reference_metrics = {}
        
#         # Historique
#         self.history = {
#             'costs': [self.reference_cost],
#             'rewards': [],
#             'actions': [],
#             'widths': [self.current_widths.copy()]
#         }
        
#         observation = self._get_observation()
#         info = self._get_info()
        
#         return observation, info
    
#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         """
#         Effectue une action (comme dans Gymnasium)
        
#         Args:
#             action: Vecteur d'actions ∈ [-1, 1]^N
        
#         Returns:
#             observation: Nouvel état
#             reward: Récompense
#             terminated: Episode terminé (succès)
#             truncated: Episode tronqué (échec/timeout)
#             info: Infos supplémentaires
#         """
#         self.step_count += 1
        
#         # ============ 1. Appliquer l'action ============
        
#         # Rescaler l'action : [-1, 1] → changement relatif
#         # action = 0 → pas de changement
#         # action = 1 → +action_scale (ex: +20%)
#         # action = -1 → -action_scale (ex: -20%)
#         delta = action * self.action_scale
        
#         # Calculer les nouvelles largeurs
#         # width_new = width_old * (1 + delta)
#         multipliers = 1.0 + delta
#         new_widths = self.current_widths * multipliers
        
#         # Clipper pour respecter les limites physiques
#         original_widths = self.modifier.get_original_widths()
#         new_widths = np.clip(
#             new_widths,
#             np.array(original_widths) * self.width_min,
#             np.array(original_widths) * self.width_max
#         )
        
#         # ============ 2. Évaluer le nouveau design ============
        
#         try:
#             results = self.objective.evaluate(new_widths)
#             cost = results['cost']
#             simulation_failed = False
            
#         except Exception as e:
#             # Simulation échouée → forte pénalité
#             if self.verbose:
#                 print(f"⚠️  Simulation failed at step {self.step_count}: {e}")
            
#             cost = self.reference_cost * 2.0  # Pénalité
#             results = {}
#             simulation_failed = True
        
#         # ============ 3. Calculer la récompense ============
        
#         # Récompense de base : amélioration du coût
#         if self.reference_cost > 0:
#             cost_improvement = (self.reference_cost - cost) / self.reference_cost
#         else:
#             cost_improvement = 0.0
        
#         # Reward = amélioration (positif si on diminue le coût)
#         reward = cost_improvement * 10.0  # Scaler pour rendre plus lisible
        
#         # Bonus si nouveau meilleur
#         if cost < self.best_cost:
#             reward += 5.0
#             self.best_cost = cost
#             self.best_widths = new_widths.copy()
            
#             if self.verbose:
#                 print(f"✨ Step {self.step_count}: Nouveau meilleur! "
#                       f"Cost={cost:.4f} (Δ={cost_improvement*100:.1f}%)")
        
#         # Pénalité si simulation échoue
#         if simulation_failed:
#             reward = -10.0
        
#         # Pénalité légère pour encourager l'efficacité
#         reward -= 0.1
        
#         # ============ 4. Déterminer la terminaison ============
        
#         terminated = False  # Succès
#         truncated = False   # Échec/timeout
        
#         # Succès : coût < seuil
#         if cost < self.reference_cost * self.success_threshold:
#             terminated = True
#             reward += 20.0  # Gros bonus pour succès
#             if self.verbose:
#                 print(f"🎉 Succès! Cost={cost:.4f} < {self.reference_cost * self.success_threshold:.4f}")
        
#         # Échec : simulation échouée
#         if simulation_failed:
#             truncated = True
        
#         # Timeout : max steps
#         if self.step_count >= self.max_steps:
#             truncated = True
#             if self.verbose:
#                 print(f"⏱️  Timeout après {self.max_steps} steps")
        
#         # ============ 5. Mise à jour de l'état ============
        
#         if not simulation_failed:
#             self.current_widths = new_widths
        
#         # Historique
#         self.history['costs'].append(cost)
#         self.history['rewards'].append(reward)
#         self.history['actions'].append(action.copy())
#         self.history['widths'].append(self.current_widths.copy())
        
#         observation = self._get_observation()
#         info = self._get_info(cost, results)
        
#         return observation, reward, terminated, truncated, info
    
#     def _get_observation(self) -> np.ndarray:
#         """
#         Construit l'observation (comme sensor dans le TP)
        
#         Returns:
#             obs: [delay_norm, power_norm, area_norm, w1_norm, ..., wN_norm]
#         """
#         # Évaluer l'état actuel
#         try:
#             results = self.objective.evaluate(self.current_widths)
            
#             # Normaliser les métriques par rapport à la référence
#             delay_norm = results.get('delay_avg', 1.0) / self.reference_metrics.get('delay_avg', 1.0)
#             power_norm = results.get('power_avg', 1.0) / self.reference_metrics.get('power_avg', 1.0)
#             area_norm = results.get('area', 1.0) / self.reference_metrics.get('area', 1.0)
            
#         except:
#             # Valeurs par défaut si échec
#             delay_norm = 1.0
#             power_norm = 1.0
#             area_norm = 1.0
        
#         # Normaliser les largeurs
#         original_widths = np.array(self.modifier.get_original_widths())
#         widths_norm = self.current_widths / original_widths
        
#         # Observation finale
#         obs = np.concatenate([
#             [delay_norm, power_norm, area_norm],
#             widths_norm
#         ]).astype(np.float32)
        
#         return obs
    
#     def _get_info(self, cost: float = None, results: Dict = None) -> Dict:
#         """Infos supplémentaires (pour debug)"""
#         info = {
#             'step': self.step_count,
#             'episode': self.episode_count,
#             'best_cost': self.best_cost,
#             'reference_cost': self.reference_cost,
#         }
        
#         if cost is not None:
#             info['current_cost'] = cost
        
#         if results:
#             info['metrics'] = results
        
#         if self.best_widths is not None:
#             info['best_widths'] = self.best_widths.tolist()
        
#         return info
    
#     def render(self):
#         """Affichage human-readable (optionnel)"""
#         if not self.verbose:
#             return
        
#         print(f"\n{'='*60}")
#         print(f"Episode {self.episode_count} - Step {self.step_count}/{self.max_steps}")
#         print(f"{'='*60}")
#         print(f"Current cost: {self.history['costs'][-1]:.4f}")
#         print(f"Best cost: {self.best_cost:.4f}")
#         print(f"Improvement: {(1 - self.best_cost/self.reference_cost)*100:.1f}%")
        
#         if len(self.history['rewards']) > 0:
#             print(f"Last reward: {self.history['rewards'][-1]:.2f}")
#             print(f"Total reward: {sum(self.history['rewards']):.2f}")
    
#     def close(self):
#         """Nettoyage (optionnel)"""
#         pass
    
#     def get_episode_summary(self) -> Dict:
#         """Résumé de l'épisode (pour analyse)"""
#         return {
#             'n_steps': self.step_count,
#             'best_cost': self.best_cost,
#             'best_widths': self.best_widths.tolist() if self.best_widths is not None else [],
#             'improvement_%': (1 - self.best_cost / self.reference_cost) * 100,
#             'total_reward': sum(self.history['rewards']),
#             'costs': self.history['costs'],
#             'rewards': self.history['rewards'],
#         }

# src/environment/gym_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Optional, Tuple
from ..optimization.objective import ObjectiveFunction
from ..simulation.netlist_generator import SimulationConfig
from ..simulation.pdk_manager import PDKManager


class StandardCellEnv(gym.Env):
    """
    Environnement Gym pour optimisation de standard cells
    
    État (observation):
        - Largeurs normalisées des transistors
        - Métriques normalisées (delay, energy, area)
    
    Action (continue):
        - Deltas de largeurs: [-0.2, +0.2] pour chaque transistor
    
    Récompense:
        - Réduction du coût multi-objectif
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        cell_name: str,
        pdk: PDKManager,
        config: SimulationConfig = None,
        cost_weights: Dict[str, float] = None,
        max_steps: int = 50,
        use_cache: bool = True,
        verbose: bool = False
    ):
        super().__init__()
        
        self.cell_name = cell_name
        self.pdk = pdk
        self.config = config or SimulationConfig()
        self.cost_weights = cost_weights or {'delay': 0.5, 'energy': 0.3, 'area': 0.2}
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Fonction objectif
        self.objective = ObjectiveFunction(
            cell_name=cell_name,
            config=self.config,
            pdk=pdk,
            verbose=verbose,
            use_cache=use_cache
        )
        
        # Largeurs originales
        self.original_widths = self.objective._get_original_widths()
        self.transistor_names = sorted(self.original_widths.keys())
        self.n_transistors = len(self.transistor_names)
        
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
        
        # Action: deltas de largeurs [-0.2, +0.2] (20% de variation)
        self.action_space = spaces.Box(
            low=-0.2,
            high=0.2,
            shape=(self.n_transistors,),
            dtype=np.float32
        )
        
        # État interne
        self.current_widths = None
        self.step_count = 0
        self.best_cost = float('inf')
        self.history = {'costs': [], 'rewards': [], 'widths': []}
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Réinitialise l'environnement"""
        super().reset(seed=seed)
        
        # Partir des largeurs originales
        self.current_widths = self.original_widths.copy()
        self.step_count = 0
        self.best_cost = float('inf')
        self.history = {'costs': [], 'rewards': [], 'widths': []}
        
        # Évaluer l'état initial
        cost, metrics = self.objective.evaluate(
            self.current_widths,
            self.cost_weights
        )
        
        self.best_cost = cost
        
        observation = self._get_observation(metrics)
        info = {'cost': cost, 'metrics': metrics}
        
        if self.verbose:
            print(f"\n🔄 Reset: cost initial = {cost:.4f}")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Applique une action (deltas de largeurs)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # 1. Calculer les nouvelles largeurs
        new_widths = {}
        for i, name in enumerate(self.transistor_names):
            original_w = self.original_widths[name]
            delta_fraction = action[i]  # [-0.2, +0.2]
            
            # Appliquer le delta
            new_w = self.current_widths[name] * (1 + delta_fraction)
            
            # Clipper aux contraintes DRC
            new_w = np.clip(new_w, self.min_width, self.max_width)
            
            new_widths[name] = new_w
        
        # 2. Évaluer la nouvelle configuration
        cost, metrics = self.objective.evaluate(new_widths, self.cost_weights)
        
        # 3. Calculer la récompense
        if cost == float('inf'):
            # Simulation échouée
            reward = -10.0
            terminated = False
            truncated = True
        else:
            # Récompense = amélioration du coût
            improvement = self.best_cost - cost
            reward = improvement * 100  # Amplifier
            
            # Bonus si nouveau meilleur
            if cost < self.best_cost:
                reward += 5.0
                self.best_cost = cost
            
            # Pénalité pour stagnation
            reward -= 0.1
            
            # Terminaison si très bon
            terminated = (cost < 0.8)  # 20% meilleur que référence
            truncated = False
        
        # 4. Mise à jour de l'état
        if cost != float('inf'):
            self.current_widths = new_widths
        
        # Historique
        self.history['costs'].append(cost)
        self.history['rewards'].append(reward)
        self.history['widths'].append(list(new_widths.values()))
        
        # Timeout
        if self.step_count >= self.max_steps:
            truncated = True
        
        observation = self._get_observation(metrics)
        info = {
            'cost': cost,
            'metrics': metrics,
            'widths': new_widths,
            'improvement': improvement if cost != float('inf') else 0
        }
        
        if self.verbose and self.step_count % 10 == 0:
            print(f"  Step {self.step_count}: cost={cost:.4f}, reward={reward:.2f}")
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self, metrics: Dict) -> np.ndarray:
        """Construit le vecteur d'observation"""
        # Normaliser les largeurs par rapport aux originales
        widths_norm = [
            self.current_widths[name] / self.original_widths[name]
            for name in self.transistor_names
        ]
        
        # Métriques normalisées
        delay_norm = metrics.get('delay_norm', 1.0)
        energy_norm = metrics.get('energy_norm', 1.0)
        area_norm = metrics.get('area_norm', 1.0)
        
        obs = np.array(
            widths_norm + [delay_norm, energy_norm, area_norm],
            dtype=np.float32
        )
        
        return obs
