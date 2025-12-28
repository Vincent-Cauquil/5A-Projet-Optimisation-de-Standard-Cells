# I4-COMSC-Projet
Projet de conception IA

# Installation 

# 1. Créer l'environnement (Python 3.12 sera téléchargé si nécessaire)
uv venv --python 3.12

# 2. Activer l'environnement
# Sur macOS/Linux :
source .venv/bin/activate
# Sur Windows :
# .venv\Scripts\activate

# 3. Installer les dépendances du projet
uv sync

# 4. Installer la librairie pyngs manuellement
uv pip install ./libs/pyngs-0.0.2-cp312-cp312-linux_x86_64.whl

# 5. Télécharger le PDK (pas besoin de 'uv run' si l'env python est activé) (step 2)
(uv run) python -m ciel ls-remote --pdk sky130
(uv run) python -m ciel enable --pdk sky130 54435919abffb937387ec956209f9cf5fd2dfbee
(uv run) python -m ciel enable --pdk sky130 54435919abffb937387ec956209f9cf5fd2dfbee