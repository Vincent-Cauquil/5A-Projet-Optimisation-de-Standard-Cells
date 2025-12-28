import sys
from pathlib import Path
import pyqtgraph as pg
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, 
                             QTabWidget, QPushButton, QSpinBox, QGroupBox, 
                             QFormLayout, QMessageBox, QDoubleSpinBox, QFrame,
                             QHeaderView, QSplitter)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QFont, QAction

# Imports locaux
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.weight_manager import WeightManager
from src.gui.workers import TrainingWorker
from src.simulation.pdk_manager import PDKManager

# === STYLE GLOBAL (Inspiré de QDAC_app) ===
APP_STYLE = """
/* RESET GLOBAL */
* {
    outline: none; /* Supprime le pointillé de focus moche */
}

QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Arial';
    font-size: 14px;
    border: none; /* Supprime les bordures blanches fantômes */
}

/* GROUPBOX (Le coupable habituel) */
QGroupBox {
    border: 2px solid #333; /* Bordure sombre explicite */
    border-radius: 6px;
    margin-top: 24px; /* Espace pour le titre */
    background-color: #1e1e1e;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #1e1e1e; /* Fond du texte pour cacher la ligne */
    color: #00aa88; /* Teal QDAC */
}

/* BOUTONS */
QPushButton {
    background-color: #2b2b2b;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 8px 16px;
    color: #fff;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #383838;
    border-color: #00aa88;
}

QPushButton:pressed {
    background-color: #00aa88;
    color: #fff;
}

QPushButton:disabled {
    background-color: #252525;
    border-color: #333;
    color: #555;
}

/* ARBRE (TREE WIDGET) */
QTreeWidget {
    background-color: #2b2b2b;
    border: 1px solid #333;
    border-radius: 4px;
    color: #ddd;
    outline: none;
}

QTreeWidget::item {
    padding: 4px;
}

QTreeWidget::item:selected {
    background-color: #005544; /* Vert foncé sélection */
    color: white;
    border: none;
}

QTreeWidget::item:hover {
    background-color: #333;
}

QHeaderView::section {
    background-color: #2b2b2b;
    border: none;
    color: #aaa;
    padding: 4px;
}

/* ONGLETS (TABS) */
QTabWidget::pane {
    border: 1px solid #333; /* Cadre sombre autour du contenu */
    background-color: #1e1e1e;
    border-radius: 4px;
}

QTabBar::tab {
    background: #252525;
    border: 1px solid #333;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 8px 20px;
    color: #888;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #1e1e1e;
    border-top: 2px solid #00aa88; /* Ligne verte au dessus */
    color: #fff;
    font-weight: bold;
}

QTabBar::tab:hover {
    background: #2b2b2b;
    color: #bbb;
}

/* CHAMPS DE SAISIE (SPINBOX) */
QSpinBox, QDoubleSpinBox {
    background-color: #121212;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 6px;
    color: #00cc99;
    font-weight: bold;
    selection-background-color: #00aa88;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #2b2b2b;
    border: none;
    width: 16px;
}

/* SPLITTER (La barre de redimensionnement) */
QSplitter::handle {
    background-color: #1e1e1e;
    border: none;
}

/* SCROLLBARS (Optionnel mais propre) */
QScrollBar:vertical {
    border: none;
    background: #1e1e1e;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #444;
    min-height: 20px;
    border-radius: 5px;
}
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sky130 RL Optimizer Studio")
        self.resize(1280, 800)
        
        # Application du style global
        self.setStyleSheet(APP_STYLE)
        
        # Managers
        self.wm = WeightManager()
        self.current_cell = None
        self.worker = None
        
        # UI Setup
        self.setup_ui()
        self.populate_tree()

    def setup_ui(self):
        """Construction de l'interface avec Splitter"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Utilisation d'un QSplitter pour redimensionner l'arbre et le contenu
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # === GAUCHE : ARBRE DES CELLULES ===
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Titre styled
        lbl_tree = QLabel("LIBRARY BROWSER")
        lbl_tree.setStyleSheet("color: #888; font-size: 12px; font-weight: bold; letter-spacing: 1px;")
        left_layout.addWidget(lbl_tree)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(20)
        self.tree.itemClicked.connect(self.on_cell_selected)
        left_layout.addWidget(self.tree)
        
        # Légende compacte
        legend_box = QFrame()
        legend_box.setStyleSheet("background-color: #2b2b2b; border-radius: 4px; padding: 5px;")
        legend_layout = QVBoxLayout(legend_box)
        legend_layout.setSpacing(2)
        
        def add_legend_item(color, text):
            row = QHBoxLayout()
            icon = QLabel()
            icon.setFixedSize(10, 10)
            icon.setStyleSheet(f"background-color: {color}; border-radius: 5px;")
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 11px; color: #aaa;")
            row.addWidget(icon)
            row.addWidget(lbl)
            row.addStretch()
            legend_layout.addLayout(row)
            
        add_legend_item("#00cc99", "Trained (RL Model Ready)")
        add_legend_item("#0077ff", "Weights Optimized")
        add_legend_item("#ffaa00", "Baseline Only")
        add_legend_item("#ff4444", "Unknown / Untouched")
        
        left_layout.addWidget(legend_box)
        
        # === DROITE : CONTENU (TABS) ===
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header Cellule
        self.header_frame = QFrame()
        self.header_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border: 2px solid #444;
                border-radius: 6px;
            }
        """)
        self.header_frame.setFixedHeight(60)
        header_layout = QHBoxLayout(self.header_frame)
        
        self.lbl_cell_name = QLabel("Select a cell to begin")
        self.lbl_cell_name.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.lbl_cell_name.setStyleSheet("color: #fff; border: none;")
        header_layout.addWidget(self.lbl_cell_name)
        
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #aaa; font-style: italic; border: none;")
        header_layout.addWidget(self.lbl_status)
        header_layout.addStretch()
        
        right_layout.addWidget(self.header_frame)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tab_train = self.create_training_tab()
        self.tabs.addTab(self.tab_train, "TRAINING")
        
        self.tab_infer = self.create_inference_tab()
        self.tabs.addTab(self.tab_infer, "INFERENCE")
        self.tabs.setTabEnabled(1, False)
        
        right_layout.addWidget(self.tabs)

        # Ajout au splitter
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([300, 900])
        splitter.setCollapsible(0, False)
        
        main_layout.addWidget(splitter)

    def create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(20)

        # 1. Configuration Group
        config_group = QGroupBox("TRAINING CONFIGURATION")
        grid = QFormLayout()
        grid.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.spin_steps = QSpinBox()
        self.spin_steps.setRange(1000, 200000)
        self.spin_steps.setValue(10000)
        self.spin_steps.setSingleStep(1000)
        self.spin_steps.setSuffix(" steps")
        
        self.spin_cores = QSpinBox()
        self.spin_cores.setRange(1, 16)
        self.spin_cores.setValue(4)
        self.spin_cores.setSuffix(" cores")
        
        grid.addRow("Total Timesteps:", self.spin_steps)
        grid.addRow("Parallel Workers:", self.spin_cores)
        
        # Bouton Start (Style QDAC)
        self.btn_start = QPushButton("START TRAINING SEQUENCE")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_training)
        self.btn_start.setEnabled(False)
        
        grid.addRow("", self.btn_start)
        config_group.setLayout(grid)
        layout.addWidget(config_group)

        # 2. Real-time Plot
        plot_group = QGroupBox("REAL-TIME METRICS")
        plot_layout = QVBoxLayout(plot_group)
        
        # Configuration sombre pour le graphe
        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', '#d0d0d0')
        
        self.plot_widget = pg.PlotWidget(title="Optimization Reward")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Reward')
        self.plot_widget.setLabel('bottom', 'Steps')
        
        # Courbe style néon
        self.curve_reward = self.plot_widget.plot(pen=pg.mkPen('#00aa88', width=2))
        plot_layout.addWidget(self.plot_widget)
        
        layout.addWidget(plot_group)
        return widget

    def create_inference_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        constraints_group = QGroupBox("DESIGN CONSTRAINTS")
        form = QFormLayout()
        
        self.target_delay = QDoubleSpinBox()
        self.target_delay.setRange(0, 5000)
        self.target_delay.setSuffix(" ps")
        self.target_delay.setValue(50)
        
        self.target_power = QDoubleSpinBox()
        self.target_power.setRange(0, 5000)
        self.target_power.setSuffix(" fJ")
        self.target_power.setValue(10)
        
        form.addRow("Target Delay:", self.target_delay)
        form.addRow("Max Energy:", self.target_power)
        
        self.btn_infer = QPushButton("RUN OPTIMIZATION INFERENCE")
        self.btn_infer.setFixedHeight(40)
        self.btn_infer.setStyleSheet("border-color: #0077ff;") # Bleu pour inférence
        self.btn_infer.clicked.connect(self.run_inference)
        
        form.addRow("", self.btn_infer)
        constraints_group.setLayout(form)
        layout.addWidget(constraints_group)
        
        # Result Box
        res_group = QGroupBox("OPTIMIZATION RESULTS")
        res_layout = QVBoxLayout(res_group)
        self.lbl_result = QLabel("Waiting for input parameters...")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setStyleSheet("font-size: 14px; color: #888;")
        res_layout.addWidget(self.lbl_result)
        layout.addWidget(res_group)
        
        layout.addStretch()
        return widget

    def _create_color_icon(self, color_name):
        """Création propre d'icône ronde (Style QDAC led)"""
        pixmap = QPixmap(14, 14)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Couleur principale
        c = QColor(color_name)
        painter.setBrush(c)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(1, 1, 12, 12)
        
        # Petit reflet pour effet LED
        painter.setBrush(QColor(255, 255, 255, 100))
        painter.drawEllipse(3, 3, 4, 4)
        
        painter.end()
        return QIcon(pixmap)

    def populate_tree(self):
        self.tree.clear()
        
        # 1. Scan via PDKManager
        try:
            pdk = PDKManager("sky130", verbose=False)
            all_raw_cells = pdk.list_available_cells() 
        except Exception as e:
            print(f"Erreur lecture PDK: {e}")
            all_raw_cells = []

        # 2. Groupes
        display_groups = {
            'Inverters': [], 'Buffers': [], 'NAND': [], 'NOR': [],
            'AND': [], 'OR': [], 'XOR': [], 'XNOR': [],
            'MUX': [], 'Latches': [], 'Flip-Flops': [], 'Other': []
        }

        # 3. Tri
        for cell_name in all_raw_cells:
            category = self._get_display_category(cell_name)
            if category in display_groups:
                display_groups[category].append(cell_name)
            else:
                display_groups['Other'].append(cell_name)

        # 4. Affichage
        for cat_name, cells in display_groups.items():
            if not cells: continue

            parent_item = QTreeWidgetItem(self.tree)
            parent_item.setText(0, cat_name)
            parent_item.setExpanded(False)
            # Style dossier
            parent_item.setForeground(0, QColor("#e0e0e0"))
            font = parent_item.font(0)
            font.setBold(True)
            parent_item.setFont(0, font)

            for cell_name in sorted(cells):
                item = QTreeWidgetItem(parent_item)
                item.setText(0, cell_name)
                item.setData(0, Qt.ItemDataRole.UserRole, cell_name)

                # --- STATUTS ---
                tech_cat = self.wm._get_category(cell_name)
                model_path = Path(f"data/models/{tech_cat}/{cell_name}.zip")
                weights_path = self.wm.base_dir / tech_cat / f"{cell_name}.json"
                baseline_path = Path(f"src/models/references/{tech_cat}_baseline.json")

                if model_path.exists():
                    item.setIcon(0, self._create_color_icon("#00cc99")) # Vert QDAC
                    item.setToolTip(0, "Ready for Inference")
                elif weights_path.exists():
                    item.setIcon(0, self._create_color_icon("#0077ff")) # Bleu
                elif self._is_in_baseline(cell_name, baseline_path):
                    item.setIcon(0, self._create_color_icon("#ffaa00")) # Orange
                else:
                    item.setIcon(0, self._create_color_icon("#ff4444")) # Rouge

    def _get_display_category(self, cell_name: str) -> str:
        c = cell_name.lower()
        if '__inv_' in c: return 'Inverters'
        elif '__buf_' in c or '__clkbuf_' in c: return 'Buffers'
        elif '__nand' in c: return 'NAND'
        elif '__nor' in c: return 'NOR'
        elif '__and' in c: return 'AND'
        elif '__xnor' in c: return 'XNOR'
        elif '__xor' in c: return 'XOR'
        elif '__or' in c: return 'OR'
        elif '__mux' in c: return 'MUX'
        elif '__dlx' in c or '__latch' in c: return 'Latches'
        elif '__df' in c or '__sdff' in c: return 'Flip-Flops'
        else: return 'Other'

    def _is_in_baseline(self, cell_name, baseline_path):
        if not baseline_path.exists(): return False
        try:
            import json
            with open(baseline_path, 'r') as f:
                data = json.load(f)
            return cell_name in data
        except: return False

    def on_cell_selected(self, item, col):
        cell_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not cell_name: return 
        
        self.current_cell = cell_name
        self.lbl_cell_name.setText(cell_name)
        
        cat = self.wm._get_category(cell_name)
        model_path = Path(f"data/models/{cat}/{cell_name}.zip")
        
        if model_path.exists():
            self.lbl_status.setText("Model trained & ready")
            self.lbl_status.setStyleSheet("color: #00cc99;")
            self.tabs.setTabEnabled(1, True)
        else:
            self.lbl_status.setText("No RL model found")
            self.lbl_status.setStyleSheet("color: #ff4444;")
            self.tabs.setTabEnabled(1, False)

        self.btn_start.setEnabled(True)
        
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
        self.btn_start.setText("TRAINING IN PROGRESS...")
        self.rewards_data = []
        
        self.worker.start()

    def update_plot(self, data):
        self.rewards_data.append(data['reward'])
        self.curve_reward.setData(self.rewards_data)
        if len(self.rewards_data) % 5 == 0: # Update moins fréquent pour perf
            QApplication.processEvents()

    def on_training_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("START TRAINING SEQUENCE")
        QMessageBox.information(self, "Sequence Finished", "Training Completed Successfully!")
        self.tabs.setTabEnabled(1, True) 
        self.populate_tree()

    def on_training_error(self, err_msg):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("START TRAINING SEQUENCE")
        QMessageBox.critical(self, "Sequence Error", f"Training Failed:\n{err_msg}")

    def run_inference(self):
        # Placeholder logique inférence
        delay = self.target_delay.value()
        power = self.target_power.value()
        self.lbl_result.setText(f"Inference Running for:\nDelay < {delay}ps | Power < {power}fJ")
        self.lbl_result.setStyleSheet("color: #00cc99; font-weight: bold;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Base style
    window = MainWindow()
    window.show()
    sys.exit(app.exec())