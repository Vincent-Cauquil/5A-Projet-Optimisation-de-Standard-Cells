
### Guide des Scripts de Test et Validation (Sky130 RL Optimizer)

#### Tests d'Intégration Complète

* **test_complete_stack.py** : Simule un flux complet (PDK, Runner, Mesures) en utilisant directement la bibliothèque de cellules officielle. Teste la capacité du système à instancier un composant réel et à extraire des délais (tplh/tphl) cohérents.
* **test_multiple_cells_v2.py** : Exécute une campagne de tests parallèle sur les 8 catégories de cellules supportées. Teste la robustesse du pipeline sur une collection de cellules variées et génère un rapport CSV des succès/échecs.
* **test_simulation_with_modification.py** : Valide le cycle complet "génération → simulation → modification physique → re-simulation". Teste si les modifications de dimensions () impactent les performances (Délai/Énergie) comme prévu.

#### Tests de Modèles et Chargement

* **test_models_loading.py** : Vérifie que NGSpice charge correctement les modèles de transistors `nfet_01v8` du PDK Sky130. Teste la validité des directives `.lib` via une simulation DC simple.
* **test_models_with_cell.py** : Intègre un modèle de cellule complexe avec la bibliothèque complète. Teste si les mesures de temps de propagation utilisent les bons seuils de tension ().
* **test_multi_netlist.py** : Teste l'exécution d'un pool de simulations sur différents modèles de circuits (filtres RC). Vérifie la capacité à consolider les résultats de paramètres variés dans un tableau unique.

#### Génération et Gestion de Netlists

* **test_netlist_generator.py** : Valide la création de netlists pour des cellules complexes comme le XOR2. Teste la génération des sources PWL (Piece-Wise Linear) et l'insertion correcte des commandes de mesure `.meas` pour des transitions spécifiques.
* **test_netlist_generator_v2.py** : Automatise le flux de génération et simulation pour un inverseur standard. Teste si les mesures de délai extraites sont exploitables et convertibles en picosecondes.
* **test_netlist_modifiable.py** : Vérifie l'intégrité du fichier après modification physique des transistors. Teste si le remplacement des largeurs  respecte le format Sky130 () et si les longueurs  sont préservées.
* **check_netlist_speed.py** : Analyse la précision temporelle du fichier de simulation. Teste si le pas de temps est optimal (ex: 10ps) pour équilibrer la charge de calcul et la vitesse d'exécution.

#### PDK et Infrastructure

* **test_pdk_manager.py** : Valide l'extraction de cellules (ex: XOR2) depuis le fichier CDL vers le format SPICE. Teste la récupération des ports (entrées, sorties, alim) et des informations structurelles de la cellule.
* **test_pdk_structure.py** : Diagnostique la présence des fichiers critiques du PDK Sky130. Teste l'existence des répertoires de modèles `sky130_fd_pr` et la validité des chemins d'inclusion.
* **test_sequential_pool.py** : Teste l'exécution séquentielle d'un lot de paramètres sur un template de circuit. Vérifie que le pool gère correctement les itérations sur des valeurs de résistance et capacité différentes.

#### Débogage et Analyse

* **debug_cost.py** : Inspecte chaque étape du calcul de coût RL pour une cellule. Teste la détection d'anomalies d'unités (aire gigantesque) et la validité des ratios de normalisation par rapport à la baseline.
* **debug_single_cell.py** : Analyse une cellule unique (ex: `inv_1`) en affichant les logs bruts `stdout/stderr` de NGSpice. Teste la validité syntaxique des lignes de transistors extraites.
* **test_cost_logic.py** : Compare le coût retourné par l'environnement RL à un calcul manuel théorique. Vérifie la cohérence parfaite entre les métriques simulées et le score final de l'agent.
* **test_rc.py** : Analyse de circuits passifs Résistance-Capacité via `pyngs`. Teste la précision des mesures de fréquence de coupure () avant d'attaquer les simulations actives.

#### Tests Spécialisés

* **test_scale_fix.py** : Vérifie la stratégie de l'échelle picomètre (). Teste si NGSpice interprète correctement les valeurs entières (ex: ) comme des micromètres ().
* **test_variation.py** : Teste l'impact physique d'une action RL (ex:  sur ). Vérifie si l'environnement détecte bien que la cellule devient plus rapide mais plus gourmande en énergie.
* **test_xor2.py** : Teste spécifiquement la logique et la consommation d'une porte XOR2_1. Valide l'extraction de l'énergie dynamique () et de la puissance moyenne () par transition.
* **test_generate_xor2.py** : Script rapide de génération pour la cellule XOR2_1. Teste le contenu brut de la netlist générée pour s'assurer que tous les ports sont correctement connectés.

#### Benchmarking et Entraînement

* **benchmark_pool_optimization.py** : Compare les performances entre le mode par défaut et le mode optimisé (`fast_mode`). Teste le gain de vitesse (Speedup) et le taux de succès des simulations.
* **train.py** : Pipeline principal pour lancer l'apprentissage d'un agent PPO. Teste la convergence du coût, la sauvegarde des poids JSON et la validation finale sur une cible "Performance".
* **run_inference.py** : Charge un modèle entraîné pour optimiser une cellule selon deux scénarios : Performance vs Low Power. Teste si l'agent adapte intelligemment les largeurs  selon les contraintes.
* **exemple_pyngs.py** : Démonstration de l'utilisation de `pyngs` pour piloter NGSpice. Teste le passage de paramètres dynamiques et la récupération de mesures sans fichiers intermédiaires.
* **inspect_temp_netlist.py** : Utilitaire d'inspection pour le pool séquentiel. Teste le parsing d'une simulation unique à partir d'un fichier `.cir` existant.
