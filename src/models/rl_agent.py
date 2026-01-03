# src/models/rl_agent.py

"""
Auteurs : Vincent Cauquil (vincent.cauquil@cpe.fr)
          Léonard Anselme (leonard.anselme@cpe.fr)

Date : Novembre 2025 - Janvier 2026
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import multiprocessing as mp

from src.environment.gym_env import StandardCellEnv
from pathlib import Path
import numpy as np
from .weight_manager import WeightManager
from typing import Optional, Dict, List, Tuple
import time

class TrainingCallback(BaseCallback):
    def __init__(
        self,
        weight_manager: WeightManager,
        cell_name: str,
        save_freq: int = 1000,
        verbose: bool = True,
        training_params: Optional[Dict] = None,
        max_no_improvement: int = 5000,
        min_delta: float = 1e-6
    ):
        self.verbose = verbose
        self.weight_manager = weight_manager
        self.cell_name = cell_name
        self.save_freq = save_freq
        self.training_params = training_params or {}

        self.best_cost = float('inf')
        self.best_widths = None
        self.best_metrics = None

        self.max_no_improvement = max_no_improvement
        self.steps_since_improvement = 0
        self.last_save_step = 0
        self.min_delta = min_delta

        self.total_steps_target = self.training_params.get('total_steps', float('inf'))
        self.should_stop = False


    def _on_step(self) -> bool:
        """Callback exécuté à chaque optimisation PPO"""

        # === Compteur de steps réels SB3 ===
        real_steps = self.model.num_timesteps

        # === Stop dur : limite globale d'entraînement ===
        if real_steps >= self.total_steps_target:
            self.should_stop = True
            self._save_current_best()
            if self.verbose:
                print(f"🛑 Limite steps atteinte : {real_steps}/{self.total_steps_target}")
            return False

        # === Récupération infos des N workers ===
        infos = self.locals.get("infos", None)

        # Rien reçu → on sort
        if infos is None or len(infos) == 0:
            return True

        # === BOUCLE sur TOUS les workers ===
        for info in infos:
            if not isinstance(info, dict):
                continue

            if "cost" not in info or "widths" not in info:
                continue

            cost = info["cost"]
            widths = info["widths"]
            metrics = info.get("metrics", {})

            # === Amélioration significative ===
            if cost < (self.best_cost - self.min_delta):
                self.best_cost = cost
                self.best_widths = widths.copy()
                self.best_metrics = metrics.copy()
                self.steps_since_improvement = 0

                if self.verbose:
                    print(f"✨ Nouveau meilleur coût = {cost:.4f} (step {real_steps})")

                # On continue la boucle car un autre worker peut être encore meilleur
                continue

            else:
                self.steps_since_improvement += 1

        # === Early stopping si aucune amélioration prolongée ===
        if self.steps_since_improvement >= self.max_no_improvement:
            if self.verbose:
                print(f"🛑 Early stopping : {self.max_no_improvement} steps sans amélioration")
            self.should_stop = True
            self._save_current_best()
            return False

        # === Sauvegarde périodique ===
        if real_steps >= self.last_save_step + self.save_freq:
            self._save_current_best()
            self.last_save_step = real_steps

        return True

    def _save_current_best(self):
        """Sauvegarde les meilleurs poids en conservant les hyperparamètres"""
        if self.best_widths is None:
            return

        # === CORRECTION : On part des paramètres initiaux ===
        current_info = self.training_params.copy()
        
        # On met à jour avec les valeurs dynamiques de l'instant T
        current_info.update({
            'executed_steps': self.model.num_timesteps, 
            'best_cost': float(self.best_cost),
            'convergence': 'ongoing' if not self.should_stop else 'stopped',
            'training_time_seconds': time.time() - current_info.get('start_train', time.time())
        })

        self.weight_manager.save_weights(
            cell_name=self.cell_name,
            widths=self.best_widths,
            metrics=self.best_metrics or {},
            training_info=current_info
        )

class RLAgent:

    def __init__(
        self,
        env: StandardCellEnv,
        wm: Optional[WeightManager] = None,
        weights_dir: Optional[Path] = None,
        load_pretrained: bool = False,

        # Nouvelle config parallélisme
        n_envs: Optional[int] = None,
        use_subprocess: bool = True,

        # PPO params
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: bool = False,
        max_no_improvement: int = 5000
    ):

        self.debug = debug
        self.env = env
        self.parallel = False if n_envs is None or n_envs <= 1 else True
        self.n_envs = n_envs or (mp.cpu_count() // 2)
        self.use_subprocess = use_subprocess
        self.verbose = verbose

        # === Initialisation du WeightManager ===
        if wm is not None:
            self.weight_manager = wm
        else:
            self.weight_manager = WeightManager(weights_dir, pdk_name=env.pdk_name)
            
        # === Hyperparamètres PPO ===
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.max_no_improvement = max_no_improvement

        # Création du vec_env
        self.vec_env = self._create_vec_env()

        # Créer ou charger le modèle
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.learning_rate,
            n_steps=self._compute_n_steps(),
            batch_size=self._compute_batch_size(),
            n_epochs=self._compute_n_epochs(),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=1 if verbose else 0,
            device='cpu'  # <--- CORRECTION : Force CPU pour éviter le warning
        )

    # === Crée un environnement vectorisé ===
    def _create_vec_env(self):
        if not self.parallel:
            return DummyVecEnv([lambda: self.env])

        def make_env(rank):
            def _init():
                new_env = StandardCellEnv(
                    cell_name=self.env.cell_name,
                    pdk=self.env.pdk,
                    config=self.env.config,
                    cost_weights=self.env.cost_weights,
                    max_steps=self.env.max_steps,
                    verbose=False,
                    use_cache=getattr(self.env, 'use_cache', True),
                    seed=(self.env._seed or 42) + rank
                )
                return new_env
            return _init

        env_fns = [make_env(i) for i in range(self.n_envs)]

        if self.use_subprocess:
            print(f"🤖 SubprocVecEnv avec {self.n_envs} workers")
            return SubprocVecEnv(env_fns)
        else:
            print(f"🔧 DummyVecEnv parallèle avec {self.n_envs} envs")
            return DummyVecEnv(env_fns)

    # === Ajustement PPO automatique ===
    def _compute_n_steps(self):
        if not self.parallel:
            return 2048
        total = 2048
        return max(64, total // self.n_envs)

    def _compute_batch_size(self):
        total = 2048
        return min(self.batch_size, total // 4)

    def _compute_n_epochs(self):
        if not self.parallel:
            return self.n_epochs
        return max(self.n_epochs, 20 // max(1, self.n_envs // 4))

    def train(self, total_timesteps, save_freq=1000, callback=None, savings_model=True) -> float:
        """
        Entraîne l'agent avec sauvegarde périodique des poids

        Args:
            total_timesteps: Nombre total de steps d'entraînement
            save_freq: Fréquence de sauvegarde (en steps)
            callback: Callback externe optionnel (ex: GUI) [CORRECTION AJOUTÉE]

        Returns:
            Meilleur coût obtenu
        """
        # ==== PARAMÈTRES D'ENTRAÎNEMENT POUR LE CALLBACK ====
        training_params = {
            "total_steps": total_timesteps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "n_envs": self.vec_env.num_envs,
            "start_train": time.time(),
        }

        # ==== CALLBACK INTERNE (Gestion Poids/Sauvegarde) ====
        internal_callback = TrainingCallback(
            weight_manager=self.weight_manager,
            cell_name=self.env.cell_name,
            save_freq=save_freq // (self.n_envs if self.parallel else 1),
            verbose=self.verbose,
            training_params=training_params,
            max_no_improvement = self.max_no_improvement, 
        )

        # ==== FUSION DES CALLBACKS ====
        # On crée une liste contenant le callback interne ET le callback GUI (si présent)
        callbacks_list = [internal_callback]
        if callback is not None:
            callbacks_list.append(callback)
        
        # On combine tout dans un CallbackList que SB3 sait gérer
        combined_callback = CallbackList(callbacks_list)

        # ==== MICRO-MODE SI total_timesteps < n_steps ====
        if total_timesteps < self.model.n_steps:
            self.model.n_steps = 1
            self.model.batch_size = 1
            self.model.n_epochs = 1
            rb = self.model.rollout_buffer
            rb.buffer_size = 1
            rb.n_steps = 1
            rb.reset()
            if self.verbose:
                print("⚠️  Micro-mode activé pour très court entraînement")

        # ==== LEARN ====
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=combined_callback, 
                reset_num_timesteps=True,
                progress_bar=self.verbose
            )

        except Exception as e:
            print(f"\n❌ ERREUR PENDANT L'ENTRAÎNEMENT (traceback complet) :{e}")

        # ==== POST-TRAIN ====
        training_params["best_cost"] = float(internal_callback.best_cost)
        training_params["end_train_seconds"] = time.time() - training_params["start_train"]
        training_params["convergence"] = "completed"

        internal_callback._save_current_best()

        if savings_model :
            # Sauvegarde modèle complet (.zip)
            model_dir = Path(f"data/{self.env.pdk_name}/models") / self.env.cell_category
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / f"{self.env.cell_full_name}.zip"
            self.model.save(str(model_path))

            print(f"💾 Modèle sauvegardé: {model_path}")

        return internal_callback.best_cost