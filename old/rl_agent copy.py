# src/models/rl_agent.py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
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
        verbose: int = 0,
        training_params: Optional[Dict] = None,
        max_no_improvement: int = 2000,
        min_delta: float = 1e-6
    ):
        super().__init__(verbose)
        self.weight_manager = weight_manager
        self.cell_name = cell_name
        self.save_freq = save_freq
        self.training_params = training_params or {}

        self.best_cost = float('inf')
        self.best_widths = None
        self.best_metrics = None

        self.max_no_improvement = max_no_improvement
        self.steps_since_improvement = 0
        self.min_delta = min_delta

        self.total_steps_target = self.training_params.get('total_steps', float('inf'))
        self.should_stop = False


    def _on_step(self) -> bool:
        """Callback exécuté à chaque optimisation PPO"""

        # === COMPTEUR DE STEPS RÉELS ===
        real_steps = self.model.num_timesteps

        # === STOP SI LIMITE ===
        if real_steps >= self.total_steps_target:
            self.should_stop = True
            self._save_current_best()
            if self.verbose:
                print(f"🛑 Limite atteinte : {real_steps}/{self.total_steps_target} steps réels")
            return False

        # === Récupération des infos env ===
        infos = self.locals.get('infos', [])
        if not infos:
            return True
        
        info = infos[0] if isinstance(infos, list) else infos

        if 'cost' in info and 'widths' in info:
            cost = info['cost']
            widths = info['widths']
            metrics = info.get('metrics', {})

            # === Amélioration significative ===
            if cost < (self.best_cost - self.min_delta):
                self.best_cost = cost
                self.best_widths = widths.copy()
                self.best_metrics = metrics.copy()
                self.steps_since_improvement = 0

                if self.verbose:
                    print(f"✨ Nouveau meilleur coût = {cost:.4f} (step {real_steps})")

            else:
                self.steps_since_improvement += 1

            # === Early stopping ===
            if self.steps_since_improvement >= self.max_no_improvement:
                if self.verbose:
                    print(f"🛑 Early stopping : {self.max_no_improvement} steps sans amélioration")
                self.should_stop = True
                self._save_current_best()
                return False

            # === Sauvegarde périodique ===
            if real_steps % self.save_freq == 0:
                self._save_current_best()

        return True

    def _save_current_best(self):
        """Sauvegarde les meilleurs poids"""
        if self.best_widths is None:
            return

        self.weight_manager.save_weights(
            cell_name=self.cell_name,
            widths=self.best_widths,
            metrics=self.best_metrics or {},
            training_info={
                'steps_real': self.model.num_timesteps,
                'best_cost': float(self.best_cost),
                'convergence': 'ongoing' if not self.should_stop else 'stopped'
            }
        )

class RLAgent:
    def __init__(
            self,
            env : StandardCellEnv, 
            weights_dir: Optional[Path] = None,
            load_pretrained: bool = False,
            
            # hyperparamètres PPO configurables
            learning_rate: float = 3e-4,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            
            algorithm: str = "PPO",

            # Debug / verbose
            verbose: bool = False,
        ):

        self.algorithm = algorithm
        self.verbose = verbose
        
        # Stocker TOUS les hyperparams
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        
        self.cell_name = env.cell_name
        self.cost_weights = env.cost_weights if hasattr(env, 'cost_weights') else None

        self.env = env
        self.vec_env = DummyVecEnv([lambda: env])
        if weights_dir:
            self.weight_manager = WeightManager(base_dir=weights_dir)
            weights_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.weight_manager = WeightManager() 

        self.weights_dir = weights_dir
        initial_widths = None
        
        # Créer ou charger le modèle
        if load_pretrained:
            model_path = self._find_pretrained_model()
            if model_path:
                print(f"📂 Chargement du modèle: {model_path}")
                self.model = PPO.load(str(model_path), env=self.vec_env)
                initial_widths = self._load_pretrained_weights()
            else:
                print("⚠️  Aucun modèle pré-entraîné trouvé, création d'un nouveau")
                self.model = self._create_new_model()
        else:
            self.model = self._create_new_model()
        
        # Si poids chargés, initialiser l'environnement
        if initial_widths is not None:
            self._warm_start(initial_widths)

    def _create_new_model(self):
        """Crée un nouveau modèle PPO avec les hyperparamètres stockés"""
        return PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=1 if self.verbose else 0
        )


    def _find_pretrained_model(self) -> Optional[Path]:
        """Cherche un modèle pré-entraîné pour cette cellule"""
        cell_dir = self.weight_manager.base_dir / self.cell_name
        if not cell_dir.exists():
            return None

        # Chercher le fichier .zip le plus récent
        model_files = list(cell_dir.glob("PPO_*.zip"))
        if model_files:
            return max(model_files, key=lambda p: p.stat().st_mtime)

        return None

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
        verbose: bool = False,
        max_no_improvement_steps: int = 2000
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
        if verbose:
            print(f"\n🚀 Entraînement {self.algorithm} sur {self.cell_name}")
            print(f"   Total timesteps: {total_timesteps}")
            print(f"   Cost weights: {self.cost_weights}") 
            print(f"   Sauvegarde: tous les {save_freq} steps\n")


        training_params = {
            'total_steps': total_timesteps,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'ent_coef': self.ent_coef,
            'vf_coef': self.vf_coef,
            'max_grad_norm': self.max_grad_norm,
            'n_envs': self.vec_env.num_envs,
            'cost_weights': self.cost_weights  
        }


        # Callback pour sauvegarder les meilleurs poids
        callback = TrainingCallback(
            weight_manager=self.weight_manager,
            cell_name=self.cell_name,
            save_freq=save_freq,
            verbose= 1 if verbose else 0,
            training_params=training_params,
            max_no_improvement=max_no_improvement_steps
        )

        # Entraîner
        training_params['start_train'] = time.time()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                reset_num_timesteps=True, 
                progress_bar=True
            )
        except KeyboardInterrupt:
            print("\n⚠️  Entraînement interrompu par l'utilisateur")
        except Exception as e:
            print(f"\n❌ Erreur pendant l'entraînement: {e}")

        training_params['best_cost'] = float(callback.best_cost)
        training_params['end_train_seconds'] = time.time() - training_params['start_train']
        training_params['convergence'] = 'completed'
        print(f"\n⏱️  Temps d'entraînement: {training_params['end_train_seconds']:.2f} secondes")
        print(f"   Steps effectués: {callback.n_calls}/{total_timesteps}")

        if verbose : self._print_training_summary()
        
        callback._save_current_best()
        
        # Sauvegarder le modèle
        model_dir = self.weight_manager.base_dir / self.cell_name
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"PPO_final.zip"
        self.model.save(str(model_path))
        print(f"   Modèle sauvegardé: {model_path}")
           
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
