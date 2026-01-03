Voici une proposition de `README.md` professionnel, structuré et complet. Il intègre vos instructions d'installation, l'arborescence déduite de nos échanges et la description technique des modules.

Copiez le contenu ci-dessous dans un fichier nommé **`README.md`** à la racine de votre projet.

```markdown
# I4-COMSC-Projet : Optimisation de Standard Cells par IA

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PDK](https://img.shields.io/badge/PDK-Sky130-green) ![RL](https://img.shields.io/badge/AI-Reinforcement%20Learning-orange)

Ce projet vise à automatiser le dimensionnement des transistors (NMOS/PMOS) de cellules standards (Standard Cells) en utilisant l'Apprentissage par Renforcement (RL). Il s'interface avec le PDK **SkyWater 130nm** et le simulateur **NGSpice** pour optimiser les métriques PPA (Power, Performance, Area).

## 📋 Prérequis

* **OS :** Linux (recommandé pour `pyngs`) ou macOS/Windows (avec adaptation).
* **Outils système :** `ngspice` doit être installé et accessible dans le PATH.
* **Python :** Version 3.12 gérée via `uv`.

## 🛠️ Installation

Nous utilisons **uv** pour une gestion rapide et efficace des dépendances.

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

# Activer et télécharger la version spécifique du PDK
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
2. **Entraînement :** Configurer les paramètres (Steps, Cores) et lancer le training. L'IA explore la physique de la cellule.
3. **Inférence :** Basculer sur l'onglet "Inférence", fixer vos cibles (Délai, Puissance) et laisser l'agent optimiser la cellule pour ces spécifications ("Design-to-Spec").

---

## 📂 Structure du Projet

```text
I4-COMSC-Projet/
├── data/                   # Données du PDK et Poids sauvegardés
│   └── sky130/
│       ├── models/         # Modèles RL entraînés (.zip)
│       └── weight/         # JSON de configuration et métriques
├── libs/                   # Librairies externes (pyngs .whl)
├── src/
│   ├── environment/        # Environnement Gym
│   │   └── gym_env.py
│   ├── gui/                # Interface Utilisateur PyQt6
│   │   ├── app_main.py
│   │   └── workers.py
│   ├── models/             # Logique IA & Gestion de données
│   │   ├── rl_agent.py
│   │   └── weight_manager.py
│   └── simulation/         # Cœur de simulation SPICE
│       ├── objective.py
│       └── pdk_manager.py
├── main.py                 # Point d'entrée de l'application
├── pyproject.toml          # Configuration des dépendances (uv)
└── README.md               # Documentation

```

---

## 🧠 Architecture et Classes Principales

Le projet est divisé en 4 modules fonctionnels.

### 1. Interface Graphique (`src/gui`)

* **`MainWindow` (`app_main.py`)** : Fenêtre principale PyQt6. Gère l'affichage des graphiques temps réel (Loss/Reward), la configuration des hyperparamètres et la sélection des cellules.
* **`TrainingWorker` / `InferenceWorker` (`workers.py`)** : Classes héritant de `QThread`. Elles exécutent les calculs lourds (Apprentissage et Simulation) en arrière-plan pour ne pas figer l'interface.

### 2. Intelligence Artificielle (`src/models`)

* **`RLAgent` (`rl_agent.py`)** : Wrapper autour de **Stable-Baselines3**. Implémente l'algorithme **PPO** (Proximal Policy Optimization). Gère la création des vecteurs d'environnements (multiprocessing).
* **`WeightManager` (`weight_manager.py`)** : Système de persistance. Sauvegarde non seulement le modèle neuronal, mais aussi toute la configuration utilisateur (VDD, Temp, Targets) dans un JSON pour assurer la reproductibilité.

### 3. Environnement RL (`src/environment`)

* **`StandardCellEnv` (`gym_env.py`)** : Environnement compatible Gymnasium.
* **Observation :** Dimensions actuelles + Métriques mesurées + Cibles.
* **Action :** Variation relative (%) des largeurs de transistors.
* **Reward Function (V1.2) :** Utilise une erreur quadratique pour punir les écarts, pénalise les incohérences physiques () et récompense le respect des tolérances. Gère aussi la pénalité anti-crash SPICE.



### 4. Simulation Core (`src/simulation`)

* **`NetlistGenerator` (`objective.py`)** : Analyse la cellule, injecte les paramètres `.param W=...` et génère automatiquement le Testbench (sources PULSE) adapté au nombre d'entrées de la porte.
* **`SpiceRunner` (`objective.py`)** : Orchestre l'exécution de NGSpice en mode batch, gère les timeouts et le parsing des fichiers `.raw`.
* **`SimulationCache` (`objective.py`)** : Système de hachage intelligent. Si une configuration {Largeurs + VDD + Temp} a déjà été simulée, renvoie le résultat en  pour accélérer l'entraînement.

---

## 👥 Auteurs

Projet réalisé dans le cadre du module IA pour l'Embarqué (I4-COMSC).

```

```