"""
Agent RL parallélisé utilisant SubprocVecEnv
Hérite de RLAgent et override uniquement la création de l'environnement vectorisé
"""

from pathlib import Path
from typing import Optional
import multiprocessing as mp

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO

from src.models.rl_agent import RLAgent


class ParallelRLAgent(RLAgent):
    """
    Version parallélisée de RLAgent
    Accélération : 3-8x avec 8-16 CPU
    """

    def __init__(
        self,
        env,
        weights_dir: Optional[Path] = None,
        n_envs: Optional[int] = None,
        use_subprocess: bool = True,
        **kwargs  # Tous les autres params de RLAgent
    ):
        """
        Args:
            env: StandardCellEnv de base
            weights_dir: Répertoire de sauvegarde
            n_envs: Nombre d'envs parallèles (auto si None)
            use_subprocess: True=SubprocVecEnv, False=DummyVecEnv (debug)
            **kwargs: learning_rate, batch_size, etc. (passés à RLAgent)
        """
        
        # ✅ Stocker les infos de l'env AVANT vectorisation
        self._base_env = env
        self.n_envs = n_envs or self._get_optimal_n_envs()
        self.use_subprocess = use_subprocess
        
        print(f"\n🔧 Configuration parallèle:")
        print(f"   CPUs disponibles: {mp.cpu_count()}")
        print(f"   Environnements parallèles: {self.n_envs}")
        print(f"   Mode: {'SubprocVecEnv' if use_subprocess else 'DummyVecEnv'}")
        
        # ✅ Appeler le parent (qui va utiliser self.vec_env qu'on override)
        super().__init__(
            env=env,
            weights_dir=weights_dir,
            **kwargs
        )
        
        # ✅ APRÈS l'init parent, remplacer vec_env par la version parallèle
        self.vec_env = self._create_parallel_env()
        
        # ✅ Recréer le modèle avec le nouveau vec_env
        self.model = self._create_new_model_parallel()

    def _get_optimal_n_envs(self) -> int:
        """Auto-détecte le nombre optimal d'environnements"""
        cpu_count = mp.cpu_count()
        # Laisser 2 CPU pour le système
        return max(1, min(cpu_count - 2, 12))

    def _create_parallel_env(self):
        """Crée l'environnement vectorisé parallèle"""
        
        if self.n_envs == 1:
            # Mode single-env (pas de parallélisation)
            return DummyVecEnv([lambda: self._base_env])
        
        print(f"🚀 Création de {self.n_envs} environnements parallèles...")
        
        # ✅ Créer les fonctions factory
        env_fns = [self._make_env_fn(i) for i in range(self.n_envs)]
        
        if self.use_subprocess:
            vec_env = SubprocVecEnv(env_fns, start_method='fork')
            print(f"   ✅ SubprocVecEnv créé ({self.n_envs} processus)")
        else:
            vec_env = DummyVecEnv(env_fns)
            print(f"   ✅ DummyVecEnv créé ({self.n_envs} envs séquentiels)")
        
        return vec_env

    def _make_env_fn(self, rank: int):
        """
        Factory pour créer un environnement avec seed unique
        
        Args:
            rank: Index de l'environnement (0 à n_envs-1)
        """
        from src.environment.gym_env import StandardCellEnv
        
        # ✅ Capturer tous les paramètres
        cell_name = self._base_env.cell_name
        pdk = self._base_env.pdk
        config = self._base_env.config
        cost_weights = self._base_env.cost_weights
        max_steps = self._base_env.max_steps
        use_cache = getattr(self._base_env, 'use_cache', True)
        
        # ✅ Récupérer le seed de base (ou 42 par défaut)
        base_seed = getattr(self._base_env, '_seed', None)
        if base_seed is None:
            base_seed = 42
        
        def _init():
            """Crée une copie indépendante de l'environnement"""
            env = StandardCellEnv(
                cell_name=cell_name,
                pdk=pdk,
                config=config,
                cost_weights=cost_weights,
                max_steps=max_steps,
                verbose=False,  # ✅ Pas de print dans les workers
                use_cache=use_cache,
                seed=base_seed + rank  # ✅ Seed unique par worker
            )
            return env
        
        return _init


    def _create_new_model_parallel(self):
        """
        Crée un modèle PPO adapté au nombre d'environnements parallèles
        """
        
        # ✅ Adapter n_steps au nombre d'envs
        # PPO collecte n_steps * n_envs expériences par update
        total_steps_per_update = 2048
        n_steps = max(64, total_steps_per_update // self.n_envs)
        
        # ✅ Adapter batch_size
        batch_size = min(self.batch_size, total_steps_per_update // 4)
        
        # ✅ Adapter n_epochs (moins d'envs = plus d'epochs)
        n_epochs = max(self.n_epochs, 20 // max(1, self.n_envs // 4))
        
        print(f"\n🤖 Hyperparamètres PPO adaptés:")
        print(f"   n_steps: {n_steps} (par env)")
        print(f"   batch_size: {batch_size}")
        print(f"   n_epochs: {n_epochs}")
        print(f"   Total steps/update: {n_steps * self.n_envs}")
        
        model = PPO(
            "MlpPolicy",
            self.vec_env,  # ✅ Utiliser le vec_env parallèle
            learning_rate=self.learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=1 if self.verbose else 0
        )
        
        return model

    def train(
        self,
        total_timesteps: int = 10_000,
        save_freq: int = 1_000,
        log_interval: int = 1,  # ✅ Ajout du paramètre
        **kwargs
    ) -> float:
        """
        Entraîne l'agent (version parallélisée)
        
        Args:
            total_timesteps: Nombre total de steps
            save_freq: Fréquence de sauvegarde (en steps)
            log_interval: Fréquence d'affichage (en updates PPO)
            **kwargs: Paramètres additionnels pour model.learn()
        
        Returns:
            Meilleur coût obtenu
        """
        # ✅ Ajuster save_freq pour le parallélisme
        adjusted_save_freq = max(1, save_freq // self.n_envs)
        
        print(f"\n🚀 Entraînement parallèle:")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Steps par env: ~{total_timesteps // self.n_envs:,}")
        print(f"   Save freq (ajusté): {adjusted_save_freq:,}")
        print(f"   Log interval: {log_interval}")
        print(f"   Speedup théorique: ~{self.n_envs}x")
        
        # ✅ Créer le callback avec save_freq ajusté
        from src.models.rl_agent import TrainingCallback
        
        callback = TrainingCallback(
            weight_manager=self.weight_manager,
            cell_name=self.cell_name,
            save_freq=adjusted_save_freq,
            verbose=1
        )
        
        # ✅ Lancer l'entraînement
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,  
            **kwargs
        )
        
        return callback.best_cost

    def cleanup(self):
        """Ferme proprement les processus parallèles"""
        if hasattr(self.vec_env, 'close'):
            self.vec_env.close()
            print("🧹 Environnements parallèles fermés")
