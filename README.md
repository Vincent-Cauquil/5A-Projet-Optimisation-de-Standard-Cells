# I4-COMSC-Projet
Projet de conception IA

# 1. Créer l'environnement virtuel avec uv
uv venv --python 3.11

# 2. Activer l'environnement
source .venv/bin/activate  # ou .venv/Scripts/activate sur Windows

# 3. Ajouter les dépendances essentielles
uv add numpy pandas gymnasium stable-baselines3 ciel

# 4. Installer pyngs (fourni par le prof)
uv pip install pyngs-<version>.whl

# 5. Télécharger le PDK Sky130
python -m ciel download sky130
