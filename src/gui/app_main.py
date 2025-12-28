import sys
from pathlib import Path
import pyqtgraph as pg # Très performant pour le plotting temps réel
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, 
                             QTabWidget, QPushButton, QSpinBox, QGroupBox, 
                             QFormLayout, QMessageBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon

# Imports locaux
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.weight_manager import WeightManager
from src.gui.workers import TrainingWorker
from src.simulation.pdk_manager import PDKManager
# from src.environment.gym_env import StandardCellEnv
# from src.models.rl_agent import RLAgent

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sky130 RL Optimizer Studio")
        self.resize(1200, 800)
        
        self.wm = WeightManager()
        self.current_cell = None
        self.worker = None

        self.init_ui()
        self.populate_tree()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === PANNEAU GAUCHE : SÉLECTEUR ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Standard Cells")
        self.tree.itemClicked.connect(self.on_cell_selected)
        left_layout.addWidget(self.tree)
        
        # Légende
        lbl_info = QLabel("🟢: Trained | 🟠: Baseline Only | 🔴: Unknown")
        lbl_info.setStyleSheet("font-size: 10px; color: gray;")
        left_layout.addWidget(lbl_info)
        
        left_panel.setFixedWidth(250)
        main_layout.addWidget(left_panel)

        # === PANNEAU CENTRAL : TABS ===
        self.tabs = QTabWidget()
        
        # Tab 1: Training
        self.tab_train = self.create_training_tab()
        self.tabs.addTab(self.tab_train, "🏋️ Training")
        
        # Tab 2: Inference (Désactivé par défaut)
        self.tab_infer = self.create_inference_tab()
        self.tabs.addTab(self.tab_infer, "🧪 Inference")
        self.tabs.setTabEnabled(1, False)

        main_layout.addWidget(self.tabs)

    def create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Header Info
        self.lbl_cell_name = QLabel("Select a cell to begin")
        self.lbl_cell_name.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.lbl_cell_name)

        # Config Area
        config_group = QGroupBox("Training Configuration")
        form = QFormLayout()
        
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1000, 1000000)
        self.spin_steps.setValue(10000)
        self.spin_steps.setSingleStep(1000)
        
        self.spin_cores = QSpinBox()
        self.spin_cores.setRange(1, 16)
        self.spin_cores.setValue(4)
        
        form.addRow("Total Timesteps:", self.spin_steps)
        form.addRow("CPU Cores:", self.spin_cores)
        config_group.setLayout(form)
        layout.addWidget(config_group)

        # Actions
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_start.setEnabled(False)
        
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

        # Graphs (PyQtGraph)
        self.plot_widget = pg.PlotWidget(title="Training Progress (Reward)")
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True)
        self.curve_reward = self.plot_widget.plot(pen=pg.mkPen('b', width=2))
        layout.addWidget(self.plot_widget)

        return widget

    def create_inference_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("🎯 Set Constraints to Optimize"))
        
        # Sliders pour cibles
        form = QFormLayout()
        self.target_delay = QDoubleSpinBox()
        self.target_delay.setRange(0, 1000)
        self.target_delay.setSuffix(" ps")
        
        self.target_power = QDoubleSpinBox()
        self.target_power.setRange(0, 1000)
        self.target_power.setSuffix(" fJ")
        
        form.addRow("Target Delay:", self.target_delay)
        form.addRow("Target Energy:", self.target_power)
        layout.addLayout(form)
        
        btn_infer = QPushButton("Run Inference")
        btn_infer.clicked.connect(self.run_inference)
        layout.addWidget(btn_infer)
        
        self.lbl_result = QLabel("Waiting for input...")
        layout.addWidget(self.lbl_result)
        layout.addStretch()
        
        return widget

    def populate_tree(self):
        """
        Peuple l'arbre en utilisant PDKManager (Cellules brutes) 
        et WeightManager (Cellules entraînées).
        """
        self.tree.clear()
        
        # 1. On récupère TOUTES les cellules brutes via le PDKManager
        # Cela permet d'afficher en rouge celles qu'on n'a jamais touchées
        try:
            # On utilise le PDKManager passé à l'init ou on en crée un
            pdk = PDKManager("sky130", verbose=False)
            all_raw_cells = pdk.list_available_cells() # Retourne liste ["sky130_fd_sc_hd__inv_1", ...]
        except Exception as e:
            print(f"Erreur lecture PDK: {e}")
            all_raw_cells = []

        # 2. On récupère les catégories depuis le WeightManager pour organiser l'arbre
        # WeightManager a déjà la logique pour dire que 'inv_1' -> catégorie 'inv'
        categories_items = {}
        
        # Création des dossiers parents dans l'arbre
        unique_cats = sorted(set(self.wm.CELL_CATEGORIES.values()))
        for cat in unique_cats:
            cat_item = QTreeWidgetItem(self.tree)
            cat_item.setText(0, cat.upper())
            cat_item.setExpanded(True) # Ouvrir par défaut
            categories_items[cat] = cat_item
        
        # Dossier "Other" pour ce qui ne match pas
        other_item = QTreeWidgetItem(self.tree)
        other_item.setText(0, "OTHER")
        categories_items["other"] = other_item

        # 3. Remplissage des items
        for cell_name in all_raw_cells:
            # On détermine la catégorie via WeightManager
            cat = self.wm._get_category(cell_name)
            parent = categories_items.get(cat, other_item)
            
            item = QTreeWidgetItem(parent)
            item.setText(0, cell_name)
            item.setData(0, Qt.ItemDataRole.UserRole, cell_name)
            
            # --- LOGIQUE DES STATUTS (COULEURS) ---
            
            # Chemins des fichiers
            # Note: WeightManager sauvegarde dans data/weight/[cat]/[cell].json
            weights_path = self.wm.base_dir / cat / f"{cell_name}.json"
            
            # Le modèle RL (.zip) est sauvegardé par l'agent dans data/models/[cat]/...
            # On suppose ici une structure standard. Il faudra peut-être ajuster train.py
            # pour qu'il sauvegarde bien dans des sous-dossiers catégories.
            model_path = Path(f"data/models/{cat}/{cell_name}.zip")
            
            # Baseline (JSON généré par generate_baselines.py)
            baseline_path = Path(f"src/models/references/{cat}_baseline.json")

            if model_path.exists():
                # VERT: Modèle RL entraîné et prêt pour inférence
                item.setIcon(0, self._create_color_icon("green"))
                item.setToolTip(0, "✅ Agent RL entraîné disponible")
                
            elif weights_path.exists():
                # BLEU: On a des poids optimisés (peut-être manuellement ou ancien run)
                item.setIcon(0, self._create_color_icon("blue"))
                item.setToolTip(0, "💾 Poids optimisés existants")
                
            elif self._is_in_baseline(cell_name, baseline_path):
                # ORANGE: On a juste la baseline (dimensions d'origine)
                item.setIcon(0, self._create_color_icon("orange"))
                item.setToolTip(0, "⚠️ Baseline uniquement (Non optimisé)")
                
            else:
                # ROUGE: Cellule inconnue (jamais simulée)
                item.setIcon(0, self._create_color_icon("red"))
                item.setToolTip(0, "❌ Jamais caractérisé")

    def _is_in_baseline(self, cell_name, baseline_path):
        """Helper rapide pour vérifier si une cellule est dans le gros JSON de baseline"""
        if not baseline_path.exists():
            return False
        try:
            # Pour éviter de recharger le JSON 100 fois, l'idéal serait de le charger
            # une fois au début de populate_tree, mais c'est une opti prématurée ici.
            import json
            with open(baseline_path, 'r') as f:
                data = json.load(f)
            return cell_name in data
        except:
            return False

    def _create_color_icon(self, color_name):
        pixmap = QColor(color_name)
        icon = QIcon()
        icon.addPixmap(pixmap.toRgb().name()) # Simplifié, nécessite QPixmap
        # En vrai PyQt:
        # p = QPixmap(10, 10)
        # p.fill(QColor(color_name))
        # return QIcon(p)
        return QIcon() # Placeholder

    def on_cell_selected(self, item, col):
        cell_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not cell_name: return # C'est une catégorie
        
        self.current_cell = cell_name
        self.lbl_cell_name.setText(f"Cell: {cell_name}")
        self.btn_start.setEnabled(True)
        
        # Vérifier si modèle existe pour activer Inference
        cat = self.wm._get_category(cell_name)
        model_path = Path(f"data/models/{cat}/{cell_name}.zip")
        self.tabs.setTabEnabled(1, model_path.exists())
        
        # Reset graph
        self.rewards_data = []
        self.curve_reward.setData([], [])

    def start_training(self):
        if not self.current_cell: return
        
        config = {
            'steps': self.spin_steps.value(),
            'cores': self.spin_cores.value(),
            'learning_rate': 3e-4
        }
        
        self.worker = TrainingWorker(self.current_cell, config)
        self.worker.signals.step_update.connect(self.update_plot)
        self.worker.signals.finished.connect(self.on_training_finished)
        self.worker.signals.error.connect(self.on_training_error)
        
        self.btn_start.setEnabled(False)
        self.btn_start.setText("Training Running...")
        self.rewards_data = []
        
        self.worker.start()

    def update_plot(self, data):
        """Reçoit les données du thread et met à jour le graph"""
        self.rewards_data.append(data['reward'])
        self.curve_reward.setData(self.rewards_data)
        
        # On pourrait aussi mettre à jour des Labels de Delay/Power ici
        if len(self.rewards_data) % 10 == 0:
            QApplication.processEvents()

    def on_training_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Start Training")
        QMessageBox.information(self, "Success", "Training Completed!")
        self.tabs.setTabEnabled(1, True) # Active Inference
        self.populate_tree() # Refresh icones

    def on_training_error(self, err_msg):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Start Training")
        QMessageBox.critical(self, "Error", f"Training Failed:\n{err_msg}")

    def run_inference(self):
        # Ici on appellerait une logique similaire à run_inference.py
        # mais idéalement dans un thread aussi (InferenceWorker)
        # Pour l'instant, simple log
        delay = self.target_delay.value()
        power = self.target_power.value()
        self.lbl_result.setText(f"Optimizing for {delay}ps / {power}fJ...\n(Logic to be linked to RLAgent.predict)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Style Fusion pour un look pro sombre/moderne
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())