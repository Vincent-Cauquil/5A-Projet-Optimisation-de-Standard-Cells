from PyQt6.QtCore import QThread
from src.gui.utils.bridge import QTSignalHandler, GuiCallback
from src.environment.gym_env import StandardCellEnv
from src.models.rl_agent import RLAgent
from src.simulation.pdk_manager import PDKManager
from pathlib import Path

class TrainingWorker(QThread):
    def __init__(self, cell_name, config, parent=None):
        super().__init__(parent)
        self.cell_name = cell_name
        self.config = config # Dict avec steps, lr, cores, etc.
        self.signals = QTSignalHandler()
        self.is_running = True

    def run(self):
        try:
            # 1. Setup (Similaire à train.py)
            pdk = PDKManager("sky130", verbose=False)
            
            # Env Maître
            env = StandardCellEnv(
                cell_name=self.cell_name,
                pdk=pdk,
                max_steps=50,
                tolerance=0.05,
                verbose=False,
                use_cache=True
            )

            # Agent
            agent = RLAgent(
                env=env,
                weights_dir=Path("data/weight"),
                learning_rate=self.config.get('learning_rate', 3e-4),
                n_envs=self.config.get('cores', 2), # Attention au CPU
                verbose=True
            )

            # 2. Callback Bridge
            gui_callback = GuiCallback(self.signals)

            # 3. Train
            agent.train(
                total_timesteps=self.config.get('steps', 5000),
                save_freq=100,
                callback=gui_callback
            )
            
            self.signals.finished.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.signals.error.emit(str(e))

    def stop(self):
        self.is_running = False
        # Note: L'arrêt propre de SB3 en cours de route nécessite plus de logique,
        # ici on laisse le thread finir ou on kill brutalement si nécessaire.