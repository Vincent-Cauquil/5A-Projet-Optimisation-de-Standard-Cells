from stable_baselines3.common.callbacks import BaseCallback
from PyQt6.QtCore import QObject, pyqtSignal

class QTSignalHandler(QObject):
    """Objet relai pour émettre des signaux depuis un thread non-QObject"""
    step_update = pyqtSignal(dict) # reward, cost, info
    finished = pyqtSignal()
    error = pyqtSignal(str)

class GuiCallback(BaseCallback):
    """Callback SB3 qui envoie les données à l'interface"""
    def __init__(self, signal_handler: QTSignalHandler, verbose=0):
        super().__init__(verbose)
        self.sig = signal_handler

    def _on_step(self) -> bool:
        # On récupère les infos de la dernière étape
        infos = self.locals.get("infos", [{}])[0]
        reward = self.locals.get("rewards", [0])[0]
        
        data = {
            "step": self.num_timesteps,
            "reward": float(reward),
            "cost": infos.get("cost", 0.0),
            "delay": infos.get("metrics", {}).get("delay_avg", 0.0),
            "power": infos.get("metrics", {}).get("energy_dyn", 0.0)
        }
        
        # Emission du signal vers l'UI
        self.sig.step_update.emit(data)
        return True