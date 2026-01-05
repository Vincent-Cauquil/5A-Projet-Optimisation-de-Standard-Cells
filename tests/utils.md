### Tests d'Intégration Complète
- **test_complete_stack.py** - Tests d'intégration pour le pipeline complet
- **test_multiple_cells_v2.py** - Tests de configuration multi-cellule (dernière version)
- **test_simulation_with_modification.py** - Modification dynamique de la netlist pendant la simulation

### Tests de Modèles et Chargement
- **test_models_loading.py** - Validation de l'import et initialisation des modèles
- **test_models_with_cell.py** - Intégration des modèles avec les définitions de cellules
- **test_multi_netlist.py** - Gestion de plusieurs netlists

### Génération et Gestion de Netlists
- **test_netlist_generator.py** - Génération basique de netlist
- **test_netlist_generator_v2.py** - Génération de netlist améliorée
- **test_netlist_modifiable.py** - Capacités de modification de netlist
- **check_netlist_speed.py** - Benchmarking de performance

### PDK et Infrastructure
- **test_pdk_manager.py** - Fonctionnalités de gestion PDK
- **test_pdk_structure.py** - Validation de la structure PDK
- **test_sequential_pool.py** - Mise en pool séquentielle des ressources

### Débogage et Analyse
- **debug_cost.py** - Débogage du calcul de coût
- **debug_single_cell.py** - Analyse de cellule unique
- **test_cost_logic.py** - Vérification de la logique de coût
- **test_rc.py** - Analyse RC (résistance-capacitance)

### Tests Spécialisés
- **test_scale_fix.py** - Ajustements d'échelle
- **test_variation.py** - Gestion des variations de paramètres
- **test_xor2.py** - Fonctionnalité de porte XOR2
- **test_generate_xor2.py** - Génération XOR2

### Benchmarking et Entraînement
- **benchmark_pool_optimization.py** - Métriques d'optimisation du pool de ressources
- **train.py** - Pipeline d'entraînement du modèle
- **run_inference.py** - Exécution d'inférence
- **exemple_pyngs.py** - Exemple de bibliothèque PyNGS
- **inspect_temp_netlist.py** - Inspection de netlist temporaire
