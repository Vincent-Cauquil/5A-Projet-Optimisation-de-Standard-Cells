"""
Gestionnaire de sauvegarde/chargement des poids optimisés
Organise par catégorie de standard cell
"""

import sys
import os
import numpy as np
from pathlib import Path
import pyqtgraph as pg
import time
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel, 
                             QTabWidget, QPushButton, QSpinBox, QGroupBox, 
                             QFormLayout, QMessageBox, QDoubleSpinBox, QFrame,
                             QHeaderView, QSplitter, QComboBox, QScrollArea, QProgressBar)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QIcon, QPixmap, QPainter, QFont, QAction

# Imports locaux
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.weight_manager import WeightManager
from src.gui.workers import TrainingWorker, InferenceWorker
from src.simulation.pdk_manager import PDKManager

# Possibilité d'ajouter d'autres PDKs  
pdks_Items = ["sky130"] 
# === STYLE GLOBAL ===
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
        self.wm = None
        self.current_cell = None
        self.worker = None
        self.current_pdk = None
        
        # UI Setup
        self.setup_ui()

    def setup_ui(self):
        """Construction de l'interface avec Splitter"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # *============================*
        # Left Panel
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # === PDK SELECTION HEADER ===
        pdk_header = QWidget()
        pdk_layout_h = QHBoxLayout(pdk_header)
        pdk_layout_h.setContentsMargins(0, 0, 0, 10)

        lbl_pdk = QLabel("TARGET PDK :")
        lbl_pdk.setStyleSheet("color: #00aa88; font-weight: bold; font-size: 11px;")
        
        self.combo_pdk = QComboBox()
        self.combo_pdk.addItems(["-- Sélectionner --"] + pdks_Items)
        self.combo_pdk.currentIndexChanged.connect(self.on_pdk_changed)

        pdk_layout_h.addWidget(lbl_pdk)
        pdk_layout_h.addWidget(self.combo_pdk)
        

        # === ARBRE DES CELLULES ===
        lbl_tree = QLabel("Cell")
        lbl_tree.setStyleSheet("color: #888; font-size: 12px; font-weight: bold; letter-spacing: 1px;")
        
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(20)
        self.tree.itemClicked.connect(self.on_cell_selected)
        
        # Légende compacte
        legend_box = QFrame()
        legend_box.setStyleSheet("background-color: #2b2b2b; border-radius: 4px; padding: 5px;")
        legend_layout = QVBoxLayout(legend_box)
        legend_layout.setSpacing(2)
        
        def add_legend_item(color, text):
            """Ajoute une ligne de légende + status couleurs"""
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
            
        add_legend_item("#00cc99", "Entrainé (RL Ready)")
        add_legend_item("#0077ff", "Poids Optimisés")
        add_legend_item("#ffaa00", "Réferences Uniquement")
        add_legend_item("#ff4444", "Non entrainable")
        
        # Ajout des éléments au layout gauche
        left_layout.addWidget(pdk_header)
        left_layout.addWidget(lbl_tree)
        left_layout.addWidget(self.tree)      
        left_layout.addWidget(legend_box)
        
        # *============================*
        # Panel Droit
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
        
        self.lbl_cell_name  = QLabel("Choisissez une cellule")
        self.lbl_status     = QLabel("")

        self.lbl_cell_name.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.lbl_cell_name.setStyleSheet("color: #fff; border: none;")
        self.lbl_status.setStyleSheet("color: #aaa; font-style: italic; border: none;")

        header_layout.addWidget(self.lbl_cell_name)
        header_layout.addWidget(self.lbl_status)        
        
        # Tabs
        self.tabs       = QTabWidget()
        self.tab_train  = self.create_training_tab()
        self.tab_sim    = self.create_simulation_tab()
        self.tab_infer  = self.create_inference_tab()

        self.tabs.addTab(self.tab_train, "Apprentissage")
        self.tabs.addTab(self.tab_sim, "Simulation")
        self.tabs.addTab(self.tab_infer, "Inférence")
        self.tabs.setTabEnabled(2, False)
    
        right_layout.addWidget(self.header_frame)
        right_layout.addWidget(self.tabs)

        # Ajout au splitter
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([300, 900])
        splitter.setCollapsible(0, False)
        
        main_layout.addWidget(splitter)

    def create_toggle_header(self, title: str, target_widget: QWidget) -> QPushButton:
        """
        Crée un bouton standardisé qui affiche/masque target_widget.
        Gère automatiquement le changement de flèche (▶ / ▼).
        """
        # 1. Création du bouton avec l'état initial 
        btn = QPushButton(f"▶  {title}")
        btn.setCheckable(True)
        btn.setChecked(False)
        
        # 2. Style unifié 
        btn.setStyleSheet("""
            QPushButton { 
                text-align: left; 
                background: transparent; 
                border: none; 
                color: #00aa88; 
                font-weight: bold; 
                font-size: 13px;
            }
            QPushButton:hover { color: #00cc99; }
        """)
        btn.setFixedHeight(30)

        # 3. Fonction de bascule
        def on_toggle(checked):
            target_widget.setVisible(checked)
            # On change juste la flèche, on garde le titre
            arrow = "▼" if checked else "▶"
            btn.setText(f"{arrow}  {title}")
            QApplication.processEvents() # refresh UI

        # 4. Connexion
        btn.toggled.connect(on_toggle)
        
        return btn

    def create_training_tab(self):
        """Création de l'onglet Training avec Scroll Area
        """
        scroll_area = QScrollArea() # Conteneur scrollable plus safe avec les "option déroulante"
        scroll_area.setWidgetResizable(True) 
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Le widget qui contiendra tout le layout
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        
        layout = QVBoxLayout(content_widget) # On applique le layout au contenu
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(20)
        # ---------------------------------
        # 1. Configuration Panel Générale
        # Paramètres de base
        grid_basic = QFormLayout()
        grid_basic.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.training_settings = {
            'steps': {'default': 10000, 'type': int, 'suffix': ' steps', 'range': (1000, 200000), 'single_step': 1},
            'cores': {'default': 4, 'type': int, 'suffix': ' cores', 'range': (1, 16), 'single_step': 1},
            'learning_rate': {'default': 3e-4, 'type': float, 'suffix': '', 'range': (1e-6, 1e-2), 'decimals': 6, 'single_step': 1e-6},
            'tolerance': {'default': 0.15, 'type': float, 'suffix': ' %', 'range': (0.01, 0.50), 'decimals': 2, 'single_step': 0.01},
        }

        for key, setting in self.training_settings.items():
            if setting['type'] == int:
                spin = QSpinBox()
                spin.setRange(*setting['range'])
            else:
                spin = QDoubleSpinBox()
                spin.setRange(*setting['range'])
                spin.setDecimals(setting.get('decimals', 4))
            spin.setSingleStep(setting.get('single_step', 0.0001))
            spin.setValue(setting['default'])
            spin.setSuffix(setting['suffix'])
            setting['widget'] = spin
            grid_basic.addRow(f"{key.replace('_', ' ').title()}:", spin)

        # Paramètres Avancés
        self.advanced_container = QWidget()
        self.advanced_container.setVisible(False)
        self.btn_toggle_adv = self.create_toggle_header("Advanced Training Settings", self.advanced_container)
        grid_advanced = QFormLayout(self.advanced_container)
        grid_advanced.setContentsMargins(10, 0, 0, 10)
        
        self.advanced_training_settings = {
            "batch_size": {'default': 64, 'type': int, 'range': (16, 512)},
            "n_epochs": {'default': 10, 'type': int, 'range': (1, 50)},
            "gamma": {'default': 0.99, 'type': float, 'range': (0.8, 0.9999)},
            "gae_lambda": {'default': 0.95, 'type': float, 'range': (0.8, 0.9999)},
            "clip_range": {'default': 0.2, 'type': float, 'range': (0.01, 1)},
            "ent_coef": {'default': 0.01, 'type': float, 'range': (0.0, 1)},
            "vf_coef": {'default': 0.5, 'type': float, 'range': (0.0, 1)},
            "max_grad_norm": {'default': 0.5, 'type': float, 'range': (0.1, 2)}
        }

        for key, setting in self.advanced_training_settings.items():
            if setting['type'] == int:
                spin = QSpinBox()
                spin.setRange(*setting['range'])
            else:
                spin = QDoubleSpinBox()
                spin.setRange(*setting['range'])
                spin.setDecimals(4)
                spin.setSingleStep(0.01)

            spin.setValue(setting['default'])
            setting['widget'] = spin
            grid_advanced.addRow(f"{key.replace('_', ' ').title()}:", spin)
        
        # Paramètres de génération des targets
        self.targets_container = QWidget()
        self.targets_container.setVisible(False)
        self.btn_toggle_targets = self.create_toggle_header("Design Targets", self.targets_container)
        grid_targets = QFormLayout(self.targets_container)
        grid_targets.setContentsMargins(10, 0, 0, 10)
        
        self.target_widgets = {} 
        target_groups = [
            {"label": "Delay (ps)", "suffix": " ps", "keys": ["delay_min_ps", "delay_max_ps"],
                "defaults": [60.0, 200.0],"range": (1.0, 5000.0)},
            {"label": "Slew (ps)", "suffix": " ps", "keys": ["slew_min_ps", "slew_max_ps"],
                "defaults": [10.0, 100.0], "range": (1.0, 2000.0)},
            {"label": "Power (µW)", "suffix": " µW", "keys": ["power_min_uW", "power_max_uW"],
                "defaults": [1.0, 100.0], "range": (0.001, 1000.0)},
            {"label": "Area (µm²)", "suffix": " µm²", "keys": ["area_min_um2", "area_max_um2"],
                "defaults": [0.3, 3.0],"range": (0.1, 50.0)}]

        for group in target_groups:
            # Conteneur horizontal pour une ligne
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)

            # SpinBox MIN
            spin_min = QDoubleSpinBox()
            spin_min.setRange(*group['range'])
            spin_min.setDecimals(3)
            spin_min.setValue(group['defaults'][0])
            spin_min.setSuffix(" min")
            spin_min.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
            
            # SpinBox MAX
            spin_max = QDoubleSpinBox()
            spin_max.setRange(*group['range'])
            spin_max.setDecimals(3)
            spin_max.setValue(group['defaults'][1])
            spin_max.setSuffix(" max")
            spin_max.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)

            # Ajout au layout horizontal
            row_layout.addWidget(spin_min)
            row_layout.addWidget(QLabel("➜")) 
            row_layout.addWidget(spin_max)
            
            # Ajout au Formulaire Principal
            grid_targets.addRow(group['label'] + " :", row_widget)

            # Enregistrement des widgets pour récupération facile plus tard
            self.target_widgets[group['keys'][0]] = spin_min
            self.target_widgets[group['keys'][1]] = spin_max

        # Boutons Start / Stop
        self.btn_start = QPushButton("Start")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_training)
        self.btn_start.setEnabled(False)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        
        # PROGRESS BAR & TIME 
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 4px;
                text-align: center;
                color: white;
                background-color: #1e1e1e;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00aa88;
                border-radius: 3px;
            }
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m Steps (%p%)")
        self.lbl_time_remaining = QLabel("Temps restant estimé : --:--:--")
        self.lbl_time_remaining.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_time_remaining.setStyleSheet("color: #888; font-size: 12px; font-style: italic;")

        # Ajout final des sections au layout principal

        config_group = QGroupBox("Configuration de l'entrainement")
        main_config_layout = QVBoxLayout() 

        main_config_layout.addLayout(grid_basic)
        main_config_layout.addWidget(self.btn_toggle_adv)
        main_config_layout.addWidget(self.advanced_container)
        main_config_layout.addWidget(self.btn_toggle_targets)
        main_config_layout.addWidget(self.targets_container)
        main_config_layout.addSpacing(10)
        main_config_layout.addLayout(btn_layout)
        main_config_layout.addSpacing(10)
        main_config_layout.addWidget(self.progress_bar)
        main_config_layout.addWidget(self.lbl_time_remaining)

        config_group.setLayout(main_config_layout)
        layout.addWidget(config_group)

        # 2. Real-time Plot
        plot_group = QGroupBox("REAL-TIME METRICS")
        plot_layout = QVBoxLayout(plot_group)
        
        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', '#d0d0d0')

        # --- Graphique 1 : Reward (Vert) ---
        self.plot_reward = pg.PlotWidget(title="Optimization Reward")
        self.plot_reward.showGrid(x=True, y=True, alpha=0.3)
        self.plot_reward.setLabel('left', 'Reward')
        self.plot_reward.setMinimumHeight(200) 
        # On cache l'axe X du haut pour éviter la redondance visuelle
        self.plot_reward.getAxis('bottom').setStyle(showValues=False)
        self.curve_reward = self.plot_reward.plot(pen=pg.mkPen('#00aa88', width=2))
        
        # --- Graphique 2 : Loss/Cost (Rouge) ---
        self.plot_loss = pg.PlotWidget(title="Optimization Loss")
        self.plot_loss.showGrid(x=True, y=True, alpha=0.3)
        self.plot_loss.setLabel('left', 'Cost (Loss)')
        self.plot_loss.setLabel('bottom', 'Steps')
        self.plot_loss.setMinimumHeight(200)
        self.curve_loss = self.plot_loss.plot(pen=pg.mkPen('#ff4444', width=2)) 
        
        # Synchronisation des axes X
        self.plot_loss.setXLink(self.plot_reward)
        plot_layout.addWidget(self.plot_reward)
        plot_layout.addWidget(self.plot_loss)
        
        layout.addWidget(plot_group)
        return scroll_area
    
    def create_simulation_tab(self):
        """Création de l'onglet Simulation avec Scroll Area
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(20)

        self.sim_widgets = {}

        grp_env = QGroupBox("Environment Conditions")
        form_env = QFormLayout(grp_env)

        spin_vdd = QDoubleSpinBox()
        spin_vdd.setRange(0.0, 5.0); spin_vdd.setSingleStep(0.1); spin_vdd.setValue(1.8)
        spin_vdd.setSuffix(" V")
        self.sim_widgets['vdd'] = spin_vdd
        form_env.addRow("Supply Voltage (VDD):", spin_vdd)

        spin_temp = QDoubleSpinBox()
        spin_temp.setRange(-50.0, 200.0); spin_temp.setValue(27.0)
        spin_temp.setSuffix(" °C")
        self.sim_widgets['temp'] = spin_temp
        form_env.addRow("Temperature:", spin_temp)

        combo_corner = QComboBox()
        combo_corner.addItems(["tt", "ss", "ff", "sf", "fs"])
        self.sim_widgets['corner'] = combo_corner
        form_env.addRow("Process Corner:", combo_corner)

        spin_cload = QDoubleSpinBox()
        spin_cload.setRange(0.0, 1000.0); spin_cload.setValue(10.0)
        spin_cload.setSuffix(" fF")
        self.sim_widgets['cload_fF'] = spin_cload 
        form_env.addRow("Load Capacitance:", spin_cload)

        layout.addWidget(grp_env)
        grp_time = QGroupBox("Transient Analysis Setup")
        form_time = QFormLayout(grp_time)

        for key, label in [('trise_ps', 'Rise Time'), ('tfall_ps', 'Fall Time')]:
            spin = QDoubleSpinBox()
            spin.setRange(1.0, 5000.0); spin.setValue(100.0)
            spin.setSuffix(" ps")
            self.sim_widgets[key] = spin
            form_time.addRow(f"Input {label}:", spin)

        spin_dur = QDoubleSpinBox()
        spin_dur.setRange(0.1, 100.0); spin_dur.setValue(2.0)
        spin_dur.setSuffix(" ns")
        self.sim_widgets['test_duration_ns'] = spin_dur
        form_time.addRow("Total Duration:", spin_dur)
        
        spin_set = QDoubleSpinBox()
        spin_set.setRange(0.1, 100.0); spin_set.setValue(1.0)
        spin_set.setSuffix(" ns")
        self.sim_widgets['settling_time_ns'] = spin_set
        form_time.addRow("Settling Time:", spin_set)

        spin_step = QDoubleSpinBox()
        spin_step.setRange(0.1, 1000.0); spin_step.setValue(10.0)
        spin_step.setSuffix(" ps")
        self.sim_widgets['tran_step_ps'] = spin_step 
        form_time.addRow("Time Step:", spin_step)

        layout.addWidget(grp_time)

        # Convergence Options
        grp_conv = QGroupBox("SPICE Convergence Options")
        form_conv = QFormLayout(grp_conv)
        convergence_params = {# (Valeur, Décimales, Info)
            'rel_tol': (1e-3, 6, "1e-3"),   
            'abs_tol': (1e-12, 15, "1e-12"),
            'vntol':   (1e-6, 9, "1e-6"),
            'gmin':    (1e-15, 18, "1e-15")
        }

        for k, (val, dec, info) in convergence_params.items():
            spin = QDoubleSpinBox()
            spin.setDecimals(dec) # Important pour ne pas afficher 0.00
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(val / 10.0) 
            spin.setValue(val)
            self.sim_widgets[k] = spin
            form_conv.addRow(f"{k} (def: {info}):", spin)

        combo_method = QComboBox()
        combo_method.addItems(["gear", "trap", "euler"])
        self.sim_widgets['method'] = combo_method
        form_conv.addRow("Method:", combo_method)

        btn_adv = self.create_toggle_header("Advanced Convergence", grp_conv)
        grp_conv.setVisible(False)

        layout.addWidget(btn_adv)
        layout.addWidget(grp_conv)
        layout.addStretch()
        
        return scroll

    def create_inference_tab(self):
        """Création de l'onglet Inference avec Scroll Area
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(15, 20, 15, 15)
        layout.setSpacing(20)
        
        constraints_group = QGroupBox("DESIGN CONSTRAINTS & CONDITIONS")
        form = QFormLayout()
        
        # Structure : 
        # Clé UI : [Widget (placeholder), Valeur Défaut, Unité, Facteur SI, Clé Interne (Optionnel)]
        self.target_chosen = { 
            'target_delay_rise':       [None, 50.0, "ps", 1e-12, 'delay_rise'],
            'target_delay_fall':       [None, 50.0, "ps", 1e-12, 'delay_fall'],
            'target_slew_in':          [None, 10.0, "ps", 1e-12, 'condition_slew'], # Condition Entrée
            'target_slew_out_rise':    [None, 50.0, "ps", 1e-12, 'slew_out_rise'],
            'target_slew_out_fall':    [None, 50.0, "ps", 1e-12, 'slew_out_fall'],
            
            'target_power':            [None, 10.0, "µW", 1e-6,  'power_dyn'],
            'target_energy':           [None, 10.0, "fJ", 1e-15, 'energy_dyn'],
            
            'target_area':             [None, 1.0,  "µm²", 1.0,   'area_um2'],
            'target_area_performance': [None, 1.0,  "µm²", 1.0,   None], # Métrique perso si besoin
            
            'target_load_capacitance': [None, 5.0,  "fF", 1e-15, 'condition_cload'], # Condition Sortie
        }

        # Génération des SpinBoxes
        for key, conf in self.target_chosen.items():
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 100000.0)
            spin.setValue(conf[1])
            spin.setSuffix(f" {conf[2]}")
            spin.setDecimals(3)
            
            # On stocke le widget dans la liste (index 0)
            self.target_chosen[key][0] = spin
            
            # Joli Label
            label_text = key.replace('_', ' ').replace('target', '').strip().title()
            form.addRow(f"{label_text} :", spin)

        self.btn_infer = QPushButton("Lancer l'Inférence (Optimisation)")
        self.btn_infer.setFixedHeight(45)
        self.btn_infer.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_infer.setStyleSheet("background-color: #0077ff; color: white; font-weight: bold;") 
        self.btn_infer.clicked.connect(self.run_inference)
        
        form.addRow("", self.btn_infer)
        constraints_group.setLayout(form)
        layout.addWidget(constraints_group)
        
        # Zone de Résultats
        res_group = QGroupBox("RÉSULTATS DE SIMULATION SPICE")
        res_layout = QVBoxLayout(res_group)
        
        self.lbl_result = QLabel("En attente des paramètres...")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_result.setStyleSheet("""
            font-family: Consolas, monospace; 
            font-size: 13px; 
            color: #00cc99; 
            background-color: #121212; 
            padding: 10px; 
            border-radius: 4px;
        """)
        self.lbl_result.setWordWrap(True)
        
        res_layout.addWidget(self.lbl_result)
        layout.addWidget(res_group)
        layout.addStretch()
        
        return scroll

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

    def on_pdk_changed(self, index):
        self.combo_pdk.clearFocus()
        QApplication.processEvents()

        selected_pdk = self.combo_pdk.currentText()
        
        if index == 0 or selected_pdk == "-- Sélectionner --":
            # Si selection invalide, on vide tout
            self.tree.clear()
            self.lbl_cell_name.setText("Veuillez sélectionner un PDK")
            self.tabs.setEnabled(False)
            self.current_pdk = None
            self.wm = None
            return

        self.wm = WeightManager(pdk_name=selected_pdk)
        self.current_pdk = selected_pdk
        self.lbl_cell_name.setText("Chargement du PDK...")
        self.populate_tree(selected_pdk)
        self.lbl_cell_name.setText("Choisissez une cellule")

    def populate_tree(self, pdk_name):
        self.tree.clear()
        
        try:
            pdk = PDKManager(pdk_name, verbose=False)
            all_raw_cells = pdk.list_available_cells() 
        except Exception as e:
            print(f"Erreur lecture PDK {pdk_name}: {e}")
            all_raw_cells = []
            QMessageBox.warning(self, "Erreur PDK", f"Impossible de charger {pdk_name}.\nVérifiez que les fichiers existent.")
            return

        display_groups = {
            'Inverters': [], 'Buffers': [], 'NAND': [], 'NOR': [],
            'AND': [], 'OR': [], 'XOR': [], 'XNOR': [],
            'MUX': [], 'Latches': [], 'Flip-Flops': [], 'Other': []
        }

        for cell_name in all_raw_cells:
            category = self._get_display_category(cell_name)
            if category in display_groups:
                display_groups[category].append(cell_name)
            else:
                display_groups['Other'].append(cell_name)

        for cat_name, cells in display_groups.items():
            if not cells: continue

            parent_item = QTreeWidgetItem(self.tree)
            parent_item.setText(0, cat_name)
            parent_item.setExpanded(False)
            parent_item.setForeground(0, QColor("#e0e0e0"))
            font = parent_item.font(0)
            font.setBold(True)
            parent_item.setFont(0, font)

            for cell_name in sorted(cells):
                item = QTreeWidgetItem(parent_item)
                item.setText(0, cell_name)
                item.setData(0, Qt.ItemDataRole.UserRole, cell_name)

                tech_cat = self.wm._get_category(cell_name)
                model_path = Path(f"data/{pdk_name}/models/{tech_cat}/{cell_name}.zip")
                weights_path = self.wm.base_dir / tech_cat / f"{cell_name}.json"
                baseline_path = Path(f"src/models/references/{pdk_name}/{tech_cat}_baseline.json")

                if model_path.exists():
                    item.setIcon(0, self._create_color_icon("#00cc99"))
                elif weights_path.exists():
                    item.setIcon(0, self._create_color_icon("#0077ff"))
                elif self._is_in_baseline(cell_name, baseline_path):
                    item.setIcon(0, self._create_color_icon("#ffaa00"))
                else:
                    item.setIcon(0, self._create_color_icon("#ff4444"))

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
        self.tabs.setEnabled(True)
        
        # --- VERIFICATION CHEMIN AVEC PDK ---
        if self.current_pdk != None:
            cat = self.wm._get_category(cell_name)
            model_path = Path(f"data/{self.current_pdk}/models/{cat}/{cell_name}.zip")
            
            if model_path.exists():
                self.lbl_status.setText("Model trained & ready")
                self.lbl_status.setStyleSheet("color: #00cc99;")
                self.tabs.setTabEnabled(2, True)
            else:
                self.lbl_status.setText("No RL model found")
                self.lbl_status.setStyleSheet("color: #ff4444;")
                self.tabs.setTabEnabled(2, False)
  
            self.load_inference_defaults(cell_name) 
            # ================================================

        else :
            self.lbl_status.setText("Please select a PDK")
            self.lbl_status.setStyleSheet("color: #ffaa00;")
            self.tabs.setTabEnabled(2, False)

        self.btn_start.setEnabled(True)
        self.rewards_data = []
        self.loss_data = []
        self.curve_reward.setData([], [])
        self.curve_loss.setData([], [])

    def start_training(self):
        if not self.current_cell or not self.current_pdk: return

        config = {}
        
        # 1. Paramètres de base (Steps, Cores, LR...)
        # Structure: self.training_settings[key]['widget']
        for key, setting in self.training_settings.items():
            config[key] = setting['widget'].value()
        
        # 2. Paramètres Avancés (Gamma, batch_size...)
        # Structure: self.advanced_training_settings[key]['widget']
        for key, setting in self.advanced_training_settings.items():
            config[key] = setting['widget'].value()

        # 3. Target Ranges (Delay, Power, Slew, Area Min/Max)
        # Structure: self.target_widgets[key] = widget directement
        if hasattr(self, 'target_widgets'):
            for key, widget in self.target_widgets.items():
                config[key] = widget.value()

        # 4. Configuration Simulation (Onglet Simulation)
        # Structure: self.sim_widgets[key] = widget directement
        if hasattr(self, 'sim_widgets'):
            sim_conf = {}
            for key, widget in self.sim_widgets.items():
                # On gère les différents types de widgets (SpinBox ou ComboBox)
                if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    sim_conf[key] = widget.value()
                elif isinstance(widget, QComboBox):
                    sim_conf[key] = widget.currentText()
            
            # On stocke toute la config simu dans une sous-clé dédiée
            config['sim_config'] = sim_conf
        
        # 5. Initialisation du Worker
        config['pdk_name'] = self.current_pdk
        print(f"🚀 Starting training for {self.current_cell} on {self.current_pdk}")
        # print("Config:", config) 
        # Reset de la barre de progression
        total_steps = config.get('steps', 10000)
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.train_start_time = time.time() # On top le chrono
        self.lbl_time_remaining.setText("Calcul du temps restant...")

        self.worker = TrainingWorker(self.current_cell, config)
        self.worker.signals.step_update.connect(self.update_plot)
        self.worker.signals.finished.connect(self.on_training_finished)
        self.worker.signals.error.connect(self.on_training_error)
        
        # 6. Mise à jour UI
        self.btn_start.setEnabled(False)
        self.btn_start.setCursor(Qt.CursorShape.WaitCursor)
        self.tree.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_start.setText("Entrainement en cours...")
        self.tabs.setTabEnabled(2, False)  

        self.rewards_data = []
        self.loss_data = []
        self.curve_reward.setData([], []) 
        self.curve_loss.setData([], [])
    
        self.worker.start()

    def update_plot(self, data):
        # Récupération Reward
        self.rewards_data.append(data['reward'])
        self.curve_reward.setData(self.rewards_data)
        cost_val = data.get('cost', 0.0) 
        self.loss_data.append(cost_val)
        self.curve_loss.setData(self.loss_data)
        
        # Mise à jour barre de progression
        current_step = data.get('step', 0)
        self.progress_bar.setValue(current_step)
        
        # Calcul ETA (Estimated Time of Arrival)
        if current_step > 0 and hasattr(self, 'train_start_time'):
            elapsed = time.time() - self.train_start_time
            avg_time_per_step = elapsed / current_step 
            remaining_steps = self.progress_bar.maximum() - current_step # Steps restants
            remaining_seconds = int(remaining_steps * avg_time_per_step)
            time_str = self._format_time(remaining_seconds)
            self.lbl_time_remaining.setText(f"Temps restant estimé : {time_str}")

        # Refresh UI moins fréquent pour miniminer l'impact performance
        if len(self.rewards_data) % 5 == 0: 
            QApplication.processEvents()

    def stop_training(self):
        if self.worker:
            self.worker.terminate()
            self.worker = None
            self.btn_start.setEnabled(True)
            self.btn_start.setText("Lancer l'entraînement")
            self.tree.setEnabled(True)
            QMessageBox.information(self, "Intteruption !", "Entraînement arrêté par l'utilisateur.")
            self.btn_stop.setEnabled(False)

    def on_training_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Lancer l'entraînement")
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)

        time_str = self._format_time(time.time() - self.train_start_time)
        QMessageBox.information(self, "Entrainement Terminée", f"Entrainement terminé en {time_str}")
        self.tabs.setTabEnabled(2, True)   # Onglet Inférence
        self.tree.setEnabled(True)
        self.populate_tree(pdk_name=self.current_pdk)

    def _format_time(self, seconds):
        """Convertit des secondes en h/m/s"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            time_str = f"{h}h {m:02d}m {s:02d}s"
        else:
            time_str = f"{m:02d}m {s:02d}s"
        return time_str

    def on_training_error(self, err_msg):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("START TRAINING SEQUENCE")
        QMessageBox.critical(self, "Sequence Error", f"Training Failed:\n{err_msg}")

    def run_inference(self):
        if not self.current_cell or not self.current_pdk:
            QMessageBox.warning(self, "Attention", "Veuillez sélectionner une cellule et un PDK.")
            return

        constraints = {} # Cibles (Reward)
        conditions = {}  # Physique (Cload, Slew in)
        
        log_text = "Paramètres d'inférence :\n"
        
        for key, conf in self.target_chosen.items():
            widget, default, unit, scale, map_key = conf
            val_ui = widget.value()
            val_si = val_ui * scale # Conversion SI
            
            # Mapping intelligent
            if map_key == 'condition_cload':
                conditions['cload'] = val_si
            elif map_key == 'condition_slew':
                conditions['slew_in'] = val_si
            elif map_key:
                constraints[map_key] = val_si
            
            # Log visuel
            log_text += f" - {key}: {val_ui} {unit}\n"

        self.lbl_result.setText(log_text + "\n⏳ Simulation en cours... Patientez.")
        self.btn_infer.setEnabled(False)

        # Lancement Worker
        config_dict = {'constraints': constraints, 'conditions': conditions}
        
        self.inf_worker = InferenceWorker(self.current_cell, self.current_pdk, config_dict)
        self.inf_worker.signals.step_update.connect(self.on_inference_result)
        self.inf_worker.signals.error.connect(self.on_training_error)
        self.inf_worker.signals.finished.connect(lambda: self.btn_infer.setEnabled(True))
        
        self.inf_worker.start()

    def on_inference_result(self, data):
        """Callback de réception des résultats SPICE"""
        metrics = data.get("metrics", {})
        widths = data.get("widths", {})
        conditions = data.get("conditions", {})
        
        # Helper de formatage
        def fmt(val, scale=1.0, unit=""):
            return f"{val * scale:.3f} {unit}"

        report = "✅ RÉSULTATS SIMULATION SPICE\n"
        report += "="*30 + "\n"
        
        # 1. Conditions appliquées
        report += "CONDITIONS PHYSIQUES :\n"
        if 'cload' in conditions:
            report += f"  • Cload   : {fmt(conditions['cload'], 1e15, 'fF')}\n"
        if 'slew_in' in conditions:
            report += f"  • Slew In : {fmt(conditions['slew_in'], 1e12, 'ps')}\n"
            
        # 2. Transistors Optimisés
        report += "\nDIMENSIONS OPTIMISÉES :\n"
        for name, w in widths.items():
            report += f"  • {name} : {fmt(w, 1e9, 'nm')}\n"
            
        # 3. Performances Mesurées
        report += "\nPERFORMANCES MESURÉES :\n"
        if 'delay_rise' in metrics:
            report += f"  • Delay Rise : {fmt(metrics['delay_rise'], 1e12, 'ps')}\n"
        if 'delay_fall' in metrics:
            report += f"  • Delay Fall : {fmt(metrics['delay_fall'], 1e12, 'ps')}\n"
        if 'power_dyn' in metrics:
            report += f"  • Puissance  : {fmt(metrics['power_dyn'], 1e6, 'µW')}\n"
        if 'energy_dyn' in metrics:
            report += f"  • Énergie    : {fmt(metrics['energy_dyn'], 1e15, 'fJ')}\n"
        if 'area_um2' in metrics:
            report += f"  • Surface    : {metrics['area_um2']:.3f} µm²\n"

        self.lbl_result.setText(report)

    def load_inference_defaults(self, cell_name):
        """
        Charge les métriques de la baseline (JSON) et calcule l'aire
        pour pré-remplir les champs de l'onglet Inférence.
        """
        if not self.current_pdk or not hasattr(self, 'target_chosen'):
            return

        # 1. Construction du chemin
        category = self.wm._get_category(cell_name)
        baseline_path = Path(f"src/models/references/{self.current_pdk}/{category}_baseline.json")
        
        if not baseline_path.exists():
            self.lbl_result.setText("⚠️ Pas de fichier baseline trouvé.")
            return

        try:
            
            with open(baseline_path, 'r') as f:
                data = json.load(f)
            
            # Vérification de la présence de la cellule
            if cell_name not in data:
                self.lbl_result.setText(f"⚠️ Cellule {cell_name} absente du fichier baseline.")
                return

            cell_data = data[cell_name]
            metrics = cell_data.get('metrics', {})
            widths = cell_data.get('widths', {})
            lengths = cell_data.get('lengths', {})

            # 2. Calcul de l'Aire (Car elle n'est pas dans 'metrics' directement)
            # Area = Somme(W * L) * 1e12 (pour passer de m² à µm²)
            calculated_area_um2 = 0.0
            if widths and lengths:
                area_m2 = sum(widths[k] * lengths.get(k, 150e-9) for k in widths)
                calculated_area_um2 = area_m2 * 1e12

            # 3. Extraction intelligente des Slews (car il y a t1, t2, t3...)
            # On fait la moyenne de tous les slews trouvés pour avoir une valeur représentative
            def get_avg_metric(pattern):
                values = [v for k, v in metrics.items() if pattern in k]
                return float(np.mean(values)) if values else None

            val_slew_in = get_avg_metric("slew_in")
            val_slew_out_rise = get_avg_metric("slew_out_rise")
            val_slew_out_fall = get_avg_metric("slew_out_fall")

            # 4. Dictionnaire Valeurs SI (Système International)
            # On mappe les clés de l'UI vers les valeurs trouvées ou calculées
            values_si = {
                'target_delay_rise': metrics.get('tplh_avg', metrics.get('delay_avg')),
                'target_delay_fall': metrics.get('tphl_avg', metrics.get('delay_avg')),
                'target_power': metrics.get('power_avg'),
                'target_energy': metrics.get('energy_dyn'),
                'target_area': calculated_area_um2,
                'target_area_performance': calculated_area_um2,
                'target_slew_in': val_slew_in,
                'target_slew_out_rise': val_slew_out_rise,
                'target_slew_out_fall': val_slew_out_fall,
            }

            # 5. Mise à jour de l'Interface Graphique
            count = 0
            for ui_key, val_si in values_si.items():
                if val_si is not None and ui_key in self.target_chosen:
                    # Récupération de la config du widget
                    # Structure target_chosen[key] = [Widget, Default, Unit, Scale, MapKey]
                    widget = self.target_chosen[ui_key][0]
                    scale_factor = self.target_chosen[ui_key][3]
                    
                    # Conversion SI -> Unité UI (ex: 1.5e-10 / 1e-12 = 150 ps)
                    val_ui = val_si / scale_factor
                    widget.setValue(val_ui)
                    count += 1
            
            # Feedback utilisateur
            self.lbl_result.setText(f"✅ Baseline chargée ({count} paramètres mis à jour)")
            self.lbl_result.setStyleSheet("color: #00cc99; font-weight: bold;")

        except Exception as e:
            print(f"Erreur chargement baseline: {e}")
            self.lbl_result.setText(f"❌ Erreur lecture baseline: {str(e)}")
            self.lbl_result.setStyleSheet("color: #ff4444;")
            print(f"Erreur chargement baseline: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec())