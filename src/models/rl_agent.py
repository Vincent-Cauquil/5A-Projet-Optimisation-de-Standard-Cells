# src/models/rl_agent.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import numpy as np
from .weight_manager import WeightManager
from typing import Optional, Dict, List, Tuple


class TrainingCallback(BaseCallback):
    """
    Callback pour sauvegarder les meilleurs poids pendant l'entraînement
    """

    def __init__(
        self,
        weight_manager: WeightManager,
        cell_name: str,
        save_freq: int = 1000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.weight_manager = weight_manager
        self.cell_name = cell_name
        self.save_freq = save_freq

        self.best_cost = float('inf')
        self.best_widths = None
        self.best_metrics = None
        self.n_calls = 0

    def _on_step(self) -> bool:
        """Appelé à chaque step de l'environnement"""
        self.n_calls += 1

        # ✅ Gérer les envs vectorisés (infos = liste)
        infos = self.locals.get('infos', [])
        if not infos:
            return True

        # ✅ Prendre la première info si vectorisé
        if isinstance(infos, list):
            info = infos[0]
        else:
            info = infos

        if 'cost' in info and 'widths' in info:
            cost = info['cost']
            widths = info['widths']
            metrics = info.get('metrics', {})

            # Nouveau meilleur ?
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_widths = widths
                self.best_metrics = metrics

                if self.verbose:
                    print(f"  🎯 Step {self.n_calls}: Nouveau meilleur cost={cost:.4f}")

                # Sauvegarder immédiatement
                self._save_best_weights()

            # Sauvegarde périodique (backup)
            elif self.n_calls % self.save_freq == 0:
                if self.verbose >= 2:
                    print(f"  💾 Step {self.n_calls}: Sauvegarde périodique (cost={cost:.4f})")
                self._save_current_weights(widths, metrics, cost)

        return True

    def _save_best_weights(self):
        """Sauvegarde les meilleurs poids trouvés"""
        if self.best_widths is None:
            return

        # Convertir Dict[str, float] -> List[float]
        widths_list = [self.best_widths[name] for name in sorted(self.best_widths.keys())]

        # Métriques complètes
        metrics_to_save = {
            'delay_avg': self.best_metrics.get('delay_avg', 0),
            'tplh': self.best_metrics.get('tplh', self.best_metrics.get('delay_avg', 0) * 1.2),
            'tphl': self.best_metrics.get('tphl', self.best_metrics.get('delay_avg', 0) * 0.8),
            'power_avg': self.best_metrics.get('power_avg', 0),
            'energy_dyn': self.best_metrics.get('energy_dyn', 0),
            'area': self.best_metrics.get('area', 1.0),
            'reference': {}
        }

        training_info = {
            'total_steps': self.n_calls,
            'best_cost': float(self.best_cost),
            'convergence': 'ongoing'
        }

        self.weight_manager.save_weights(
            cell_name=self.cell_name,
            widths=widths_list,
            metrics=metrics_to_save,
            training_info=training_info,
            algorithm="PPO"
        )

    def _save_current_weights(self, widths: dict, metrics: dict, cost: float):
        """Sauvegarde périodique (backup)"""
        widths_list = [widths[name] for name in sorted(widths.keys())]

        metrics_to_save = {
            'delay_avg': metrics.get('delay_avg', 0),
            'power_avg': metrics.get('power_avg', 0),
            'energy_dyn': metrics.get('energy_dyn', 0),
            'area': metrics.get('area', 1.0),
        }

        training_info = {
            'total_steps': self.n_calls,
            'current_cost': float(cost),
            'convergence': 'backup'
        }

        # Sauvegarder dans un fichier temporaire
        backup_name = f"{self.cell_name}_backup"
        self.weight_manager.save_weights(
            cell_name=backup_name,
            widths=widths_list,
            metrics=metrics_to_save,
            training_info=training_info,
            algorithm="PPO"
        )


class RLAgent:
    """Agent PPO pour optimisation de standard cells avec sauvegarde"""

    def __init__(
        self,
        env,
        weights_dir: Path = None,
        learning_rate: float = 3e-4,
        algorithm: str = "PPO",
        load_pretrained: bool = False
    ):
        # ✅ Gérer env simple ou vectorisé
        if hasattr(env, 'vec_env'):
            # Environnement vectorisé (VectorizedStandardCellEnv)
            self.vec_env = env.vec_env
            self.is_vectorized = True
            self.n_envs = env.n_envs
            self.cell_name = env.cell_name
            print(f"✅ Utilisation d'environnements vectorisés ({self.n_envs} envs)")
        else:
            # Environnement simple → wrapper
            self.vec_env = DummyVecEnv([lambda: env])
            self.is_vectorized = False
            self.n_envs = 1
            self.cell_name = env.cell_name

        # ✅ Normalisation (optionnel mais recommandé)
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )

        self.env = env  # Garder référence
        self.algorithm = algorithm
        self.learning_rate = learning_rate

        # ✅ Initialiser le WeightManager
        if weights_dir:
            self.weight_manager = WeightManager(base_dir=weights_dir)
            weights_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.weight_manager = WeightManager()  # Utilise le répertoire par défaut

        self.weights_dir = weights_dir

        # Charger des poids pré-entraînés ?
        initial_widths = None
        if load_pretrained:
            initial_widths = self._load_pretrained_weights()

        # ✅ Créer le modèle PPO
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=learning_rate,
            n_steps=2048 // self.n_envs,  # ✅ Adapter au nombre d'envs
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )

        # Si poids chargés, initialiser l'environnement
        if initial_widths is not None:
            self._warm_start(initial_widths)

    def _load_pretrained_weights(self) -> Optional[Dict[str, float]]:
        """Charge des poids pré-entraînés pour warm start"""
        print(f"🔍 Recherche de poids pré-entraînés pour {self.cell_name}...")
        data = self.weight_manager.load_weights(self.cell_name)

        if data is None:
            # Essayer de charger depuis une cellule similaire
            category = self.weight_manager._get_category(self.cell_name)
            similar_cells = self.weight_manager.list_available_cells(category=category)

            if similar_cells:
                print(f"   Cellules similaires trouvées: {similar_cells}")
                # Charger la première cellule similaire
                data = self.weight_manager.load_weights(similar_cells[0])
            else:
                print("   Aucun poids pré-entraîné trouvé")
                return None

        if data:
            widths_list = data['optimized_widths']
            
            # ✅ Récupérer les noms de transistors depuis l'env réel
            if hasattr(self.env, 'original_widths'):
                transistor_names = sorted(self.env.original_widths.keys())
            else:
                # Pour env vectorisé, accéder au premier env
                transistor_names = sorted(self.vec_env.envs[0].original_widths.keys())

            # Reconstruire le dictionnaire
            if len(widths_list) == len(transistor_names):
                widths_dict = {name: width for name, width in zip(transistor_names, widths_list)}
                print(f"✅ Poids chargés: {widths_dict}")
                return widths_dict
            else:
                print(f"⚠️  Incompatibilité: {len(widths_list)} poids vs {len(transistor_names)} transistors")
                return None

        return None

    def _warm_start(self, initial_widths: Dict[str, float]):
        """Initialise l'environnement avec des largeurs pré-entraînées"""
        print("🔥 Warm start avec poids pré-entraînés")

        # ✅ Accéder au bon environnement
        if self.is_vectorized:
            target_env = self.vec_env.envs[0]
        else:
            target_env = self.env

        # Réinitialiser l'environnement
        obs, _ = target_env.reset()

        # Calculer l'action pour atteindre les largeurs cibles
        for name, target_width in initial_widths.items():
            if name in target_env.current_widths:
                target_env.current_widths[name] = target_width

        print(f"   État initial modifié: {target_env.current_widths}")

    def train(
        self,
        total_timesteps: int = 10000,
        save_freq: int = 500,
        verbose: int = 1
    ) -> float:
        """
        Entraîne l'agent avec sauvegarde périodique des poids

        Args:
            total_timesteps: Nombre total de steps d'entraînement
            save_freq: Fréquence de sauvegarde (en steps)
            verbose: 0=silent, 1=info, 2=debug

        Returns:
            Meilleur coût obtenu
        """
        if verbose > 0:
            print(f"\n🚀 Entraînement {self.algorithm} sur {self.cell_name}")
            print(f"   Total timesteps: {total_timesteps}")
            print(f"   Envs parallèles: {self.n_envs}")
            print(f"   Simulations effectives: {total_timesteps * self.n_envs}")
            print(f"   Sauvegarde: tous les {save_freq} steps\n")

        # Callback pour sauvegarder les meilleurs poids
        callback = TrainingCallback(
            weight_manager=self.weight_manager,
            cell_name=self.cell_name,
            save_freq=save_freq,
            verbose=verbose
        )

        # ✅ Entraîner
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False  # ✅ Désactiver la barre de progression
        )

        # Sauvegarder le modèle final
        if self.weights_dir:
            model_path = self.weights_dir / f"{self.cell_name}_final.zip"
            self.model.save(str(model_path))
            print(f"\n💾 Modèle final sauvegardé: {model_path}")

        if verbose > 0:
            print(f"\n✅ Entraînement terminé")
            print(f"   Meilleur coût: {callback.best_cost:.4f}")
            self._print_training_summary()

        return callback.best_cost

    def _print_training_summary(self):
        """Affiche un résumé des poids sauvegardés"""
        print("\n📊 RÉSUMÉ DES POIDS SAUVEGARDÉS")
        print("="*60)

        data = self.weight_manager.load_weights(self.cell_name)
        if data:
            print(f"Cellule: {data['cell_info']['full_name']}")
            print(f"Catégorie: {data['cell_info']['category']}")
            print(f"Transistors: {data['cell_info']['n_transistors']}")
            print(f"\nMétriques:")
            print(f"  Délai   : {data['metrics']['delay_avg_ps']:.2f} ps")
            print(f"  Puissance: {data['metrics']['power_avg_uw']:.3f} µW")
            print(f"  Énergie  : {data['metrics']['energy_dyn_fJ']:.3f} fJ")
            print(f"  Aire (rel): {data['metrics']['area_relative']:.3f}")

            print(f"\nLargeurs optimales:")
            
            # ✅ Récupérer les noms depuis le bon env
            if hasattr(self.env, 'original_widths'):
                transistor_names = sorted(self.env.original_widths.keys())
                original_widths = self.env.original_widths
            else:
                transistor_names = sorted(self.vec_env.envs[0].original_widths.keys())
                original_widths = self.vec_env.envs[0].original_widths
            
            for i, name in enumerate(transistor_names):
                if i < len(data['optimized_widths']):
                    width = data['optimized_widths'][i]
                    original = original_widths[name]
                    delta = (width - original) / original * 100
                    print(f"  {name}: {original:.0f} nm → {width:.0f} nm ({delta:+.1f}%)")

        print("="*60)

    def evaluate(
        self, 
        n_episodes: int = 10
    ) -> Tuple[float, float, List[Dict[str, float]]]:
        """
        Évalue l'agent et retourne les statistiques
        
        Returns:
            (mean_cost, std_cost, widths_history)
        """
        costs = []
        widths_history = []

        print(f"\n📊 Évaluation sur {n_episodes} épisodes...")

        for ep in range(n_episodes):
            obs = self.vec_env.reset()
            done = False
            episode_widths = None

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.vec_env.step(action)
                
                # ✅ Gérer info vectorisé
                if isinstance(info, list):
                    info = info[0]
                    done = done[0]
                
                if 'widths' in info:
                    episode_widths = info['widths'].copy()

            if 'cost' in info:
                costs.append(info['cost'])
                if episode_widths:
                    widths_history.append(episode_widths)
                
                print(f"  Épisode {ep+1}: cost={info['cost']:.4f}")

        mean_cost = np.mean(costs)
        std_cost = np.std(costs)

        print(f"\n📊 Statistiques sur {n_episodes} épisodes:")
        print(f"   Coût moyen: {mean_cost:.4f} ± {std_cost:.4f}")
        print(f"   Meilleur  : {min(costs):.4f}")
        print(f"   Pire      : {max(costs):.4f}")

        return mean_cost, std_cost, widths_history

    def save(self, path: Optional[Path] = None):
        """Sauvegarde le modèle"""
        if path is None and self.weights_dir:
            path = self.weights_dir / f"{self.cell_name}_final.zip"
        
        if path:
            self.model.save(str(path))
            print(f"💾 Modèle sauvegardé: {path}")

    def load(self, path: Path):
        """Charge un modèle sauvegardé"""
        self.model = PPO.load(str(path), env=self.vec_env)
        print(f"📥 Modèle chargé: {path}")
