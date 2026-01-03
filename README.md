# I4-COMSC-Projet : Optimisation de Standard Cells par IA

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PDK](https://img.shields.io/badge/PDK-Sky130-green) ![RL](https://img.shields.io/badge/AI-Reinforcement%20Learning-orange)

Ce projet vise à automatiser le dimensionnement des transistors (NMOS/PMOS) de cellules standards (Standard Cells) en utilisant l'Apprentissage par Renforcement (RL). Il s'interface avec le PDK **SkyWater 130nm** et le simulateur **NGSpice** pour optimiser les métriques PPA (Power, Performance, Area).

## 📋 Prérequis

* **OS :** Linux (recommandé pour `pyngs`) ou macOS/Windows.
* **Outils système :** `ngspice` doit être installé et accessible dans le PATH.
* **Python :** Version 3.12 gérée via `uv`.

## 🛠️ Installation

Nous utilisons **uv** pour une gestion rapide et reproductible des dépendances.

### 1. Configuration de l'environnement

```bash
# 1. Créer l'environnement virtuel (Python 3.12 sera téléchargé automatiquement)
uv venv --python 3.12

# 2. Activer l'environnement
# Sur macOS/Linux :
source .venv/bin/activate
# Sur Windows :
# .venv\Scripts\activate

# 3. Installer les dépendances du projet
uv sync

# 4. Installer la librairie pyngs manuellement (Interface NGSpice)
uv pip install ./libs/pyngs-0.0.2-cp312-cp312-linux_x86_64.whl

```

### 2. Installation du PDK SkyWater 130nm

Le projet utilise l'outil `ciel` pour gérer les PDKs. Exécutez ces commandes une fois l'environnement activé :

```bash
# Lister les PDKs disponibles
uv run python -m ciel ls-remote --pdk sky130

# Activer et télécharger la version spécifique du PDK utilisée pour le projet
uv run python -m ciel enable --pdk sky130 54435919abffb937387ec956209f9cf5fd2dfbee

```

---

## 🚀 Utilisation

Pour lancer l'interface graphique du studio d'optimisation :

```bash
uv run main.py

```

### Workflow typique :

1. **Sélection :** Choisir une cellule (ex: `sky130_fd_sc_hd__inv_1`) dans l'arbre à gauche.
2. **Entraînement :** Configurer les paramètres (Steps, Cores) et lancer le training. L'IA explore la physique de la cellule (Mode Exploration).
3. **Inférence :** Basculer sur l'onglet "Inférence", fixer vos cibles (Délai, Puissance) et laisser l'agent optimiser la cellule pour ces spécifications (Mode Design-to-Spec).

---

## 📂 Structure du Projet

```text
I4-COMSC-Projet/
├── data/                           # Données du PDK et Poids sauvegardés
├── libs/                           # Librairies externes (.whl pyngs)
├── netlists/                       # Fichiers SPICE temporaires ou templates
├── src/
│   ├── environment/                # Environnement Gym
│   │   └── gym_env.py
│   ├── gui/                        # Interface Utilisateur PyQt6
│   │   ├── utils/                  # Utilitaires graphiques
│   │   ├── app_main.py             # Fenêtre principale
│   │   └── workers.py              # Threads de calcul (QThread)
│   ├── models/                     # Logique IA & Gestion de données
│   │   ├── references/             # Baselines JSON pour Sky130
│   │   ├── rl_agent.py             # Wrapper PPO (Stable-Baselines3)
│   │   └── weight_manager.py       # Sauvegarde Config & Poids
│   ├── optimization/               # Algorithmes d'optimisation & Cache
│   │   ├── cell_modifier.py
│   │   ├── objective.py            # Extraction des métriques PPA
│   │   └── simulation_cache.py     # Cache de simulation (Hash)
│   └── simulation/                 # Interface Physique & SPICE
│       ├── netlist_generator.py    # Génération Testbench auto
│       ├── pdk_manager.py
│       └── spice_runner.py         # Exécution NGSpice
├── tests/                          # Scripts de tests unitaires
├── main.py                         # Point d'entrée de l'application
├── pyproject.toml                  # Configuration des dépendances (uv)
└── README.md                       # Documentation

```

---

## 🧠 Architecture et Classes Principales

Le code est modulaire, séparant l'IA, la physique et l'interface.

### 1. Interface Graphique (`src/gui`)

* **`MainWindow` (`app_main.py`)** : Gère l'affichage temps réel, la configuration des cibles et l'orchestration générale.
* **`TrainingWorker` / `InferenceWorker` (`workers.py`)** : Exécutent les tâches longues en arrière-plan pour garder l'UI fluide.

### 2. Intelligence Artificielle (`src/models`)

* **`RLAgent` (`rl_agent.py`)** : Agent PPO configuré pour des espaces d'actions continus. Gère le multiprocessing.
* **`WeightManager` (`weight_manager.py`)** : Assure la reproductibilité en sauvegardant un "snapshot" complet (Poids + Config Utilisateur + Métriques) en JSON.

### 3. Environnement RL (`src/environment`)

* **`StandardCellEnv` (`gym_env.py`)** :
* Traduit les actions de l'agent (variation %) en dimensions physiques.
* Calcule la **Reward V1.2** (Erreur quadratique + Contraintes physiques + Pénalité anti-crash).

### 4. Optimisation (`src/optimization`)

* **`Objective` (`objective.py`)** : Parse les fichiers `.raw` de NGSpice pour extraire *Delay*, *Slew*, *Power*. Calcule l'aire active.
* **`SimulationCache` (`simulation_cache.py`)** : Table de hachage stockant les résultats des simulations précédentes. Renvoie le résultat en  si la configuration {Largeurs + VDD + Temp} est connue.

### 5. Simulation (`src/simulation`)

* **`NetlistGenerator` (`netlist_generator.py`)** : Analyse la signature de la cellule (nombre d'entrées) et génère automatiquement le Testbench SPICE (sources PULSE) approprié.
* **`SpiceRunner` (`spice_runner.py`)** : Wrapper système pour NGSpice. Gère l'exécution batch et les timeouts.

---

## 👥 Auteurs

Projet réalisé dans le cadre du module **IA pour l'Embarqué (I4-COMSC)**.

```

```