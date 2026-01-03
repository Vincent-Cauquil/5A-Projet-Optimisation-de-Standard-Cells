import traceback
from PyQt6.QtCore import QThread
from src.gui.utils.bridge import QTSignalHandler, GuiCallback
from src.environment.gym_env import StandardCellEnv
from src.models.rl_agent import RLAgent
from src.simulation.pdk_manager import PDKManager
from src.models.weight_manager import WeightManager
from src.simulation.netlist_generator import SimulationConfig
from pathlib import Path
from stable_baselines3 import PPO
import numpy as np

class TrainingWorker(QThread):
    def __init__(self, cell_name, config, parent=None):
        super().__init__(parent)
        self.cell_name = cell_name
        self.config = config 
        self.pdk_name = config.get('pdk_name', 'sky130')
        self.signals = QTSignalHandler()
        self.is_running = True
        self.verbose = True

    def run(self):
        try:
            # === 1. CHEMINS DYNAMIQUES ===
            base_data_dir = Path("data") / self.pdk_name
            weights_dir = base_data_dir / "weight"
            
            # Pour l'environnement : besoin de la catégorie
            wm = WeightManager(pdk_name=self.pdk_name)

            # === 2. CONFIGURATION ENVIRONNEMENT (Simulation) ===
            pdk = PDKManager(self.pdk_name, verbose=self.verbose)
            
            # On passe ici seulement les paramètres liés à la simulation/physique
            env = StandardCellEnv(
                cell_name=self.cell_name,
                pdk=pdk,
                max_steps=self.config.get('max_steps', 50),     
                tolerance=self.config.get('tolerance', 0.05),
                verbose=self.verbose,
                use_cache=True
            )

            # === 3. CONFIGURATION AGENT (Apprentissage) ===
            agent_kwargs = {
                'learning_rate': self.config.get('learning_rate', 3e-4),
                'n_envs': self.config.get('cores', 2),
                'batch_size': self.config.get('batch_size'),
                'n_epochs': self.config.get('n_epochs'),
                'gamma': self.config.get('gamma'),
                'gae_lambda': self.config.get('gae_lambda'),
                'clip_range': self.config.get('clip_range'),
                'ent_coef': self.config.get('ent_coef'),
                'vf_coef': self.config.get('vf_coef'),
                'max_grad_norm': self.config.get('max_grad_norm'),
            }
            
            # Nettoyage : On ne garde que ceux qui sont définis (non None)
            agent_kwargs = {k: v for k, v in agent_kwargs.items() if v is not None}

            # On instancie l'agent en lui passant ces paramètres
            agent = RLAgent(
                env=env,
                weights_dir=weights_dir,
                verbose=self.verbose,
                **agent_kwargs  
            )

            # === 4. LANCEMENT ===
            gui_callback = GuiCallback(self.signals)
            gui_callback.should_stop_training = lambda: not self.is_running

            print(f"🚀 Start Training: {self.cell_name} on {self.pdk_name}")

            agent.train(
                total_timesteps=self.config.get('steps', 5000),
                save_freq=0, 
                callback=gui_callback,
                savings_model=True
            )
            
            self.signals.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def stop(self):
        """Méthode appelée quand l'utilisateur clique sur STOP"""
        self.is_running = False

class InferenceWorker(QThread):
    def __init__(self, cell_name, pdk_name, config_dict, parent=None): # config_dict au lieu de constraints
        super().__init__(parent)
        self.cell_name = cell_name
        self.pdk_name = pdk_name
        
        # On déballe la config
        self.constraints = config_dict.get('constraints', {})
        self.conditions = config_dict.get('conditions', {})
        
        self.signals = QTSignalHandler()

    def run(self):
        try:
            
            
            # 1. Chemins & Managers
            base_dir = Path("data") / self.pdk_name
            wm = WeightManager(pdk_name=self.pdk_name)
            category = wm._get_category(self.cell_name)
            model_path = base_dir / "models" / category / f"{self.cell_name}.zip"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Modèle introuvable : {model_path}")

            # 2. Configuration Simulation (Avec Cload dynamique !)
            pdk = PDKManager(self.pdk_name, verbose=False)
            
            # Création d'une config de simulation spécifique
            sim_config = SimulationConfig() # Valeurs par défaut
            
            # Application de la Cload demandée par l'utilisateur
            if 'cload' in self.conditions:
                sim_config.cload = self.conditions['cload']
                print(f"⚙️ Inférence avec Cload forcée : {sim_config.cload} F")

            # Application du Slew In si présent
            if 'slew_in' in self.constraints:
                 sim_config.trise = self.constraints['slew_in']
                 sim_config.tfall = self.constraints['slew_in']

            # Création de l'env avec la bonne config physique
            env = StandardCellEnv(
                cell_name=self.cell_name,
                pdk=pdk,
                config=sim_config, 
                max_steps=20,
                verbose=True,
                use_cache=False 
            )

            # 3. Chargement Modèle
            print(f"🔄 Chargement du modèle : {model_path}")
            model = PPO.load(model_path, env=env, device='cpu')

            # 4. Boucle Inférence
            obs, info = env.reset(options=self.constraints) 
            
            done = False
            final_metrics = {}
            final_widths = {}
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                final_metrics = info['metrics']
                final_widths = info['widths']
                done = terminated or truncated

            # 5. Résultat
            result_data = {
                "metrics": final_metrics,
                "widths": final_widths,
                "targets": self.constraints
            }
            self.signals.step_update.emit(result_data)
            self.signals.finished.emit()

        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))