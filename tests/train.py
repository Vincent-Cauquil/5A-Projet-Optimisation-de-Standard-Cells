import sys
import time
import argparse
from pathlib import Path

# Ajout du chemin racine pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.gym_env import StandardCellEnv
from src.models.rl_agent import RLAgent
from src.simulation.pdk_manager import PDKManager

def train_agent(
    cell_name: str = "sky130_fd_sc_hd__inv_1",
    total_timesteps: int = 20000,
    n_envs: int = 4,
    force_cpu: bool = False,
    save_freq: int = 100,
):
    """
    Lance l'entraînement PPO sur une cellule standard spécifique.
    
    L'agent va apprendre à satisfaire des contraintes aléatoires (Goal-Conditioned)
    générées par l'environnement (Delay, Power, Area).
    """
    print("="*80)
    print(f"🚀 DÉMARRAGE DE L'ENTRAÎNEMENT RL : {cell_name}")
    print(f"🎯 Objectif : {total_timesteps} steps sur {n_envs} workers parallèles")
    print("="*80)

    # 1. Initialisation du PDK
    # verbose=False pour éviter de spammer la console avec 4 workers
    pdk = PDKManager("sky130", verbose=False)
    
    # 2. Création de l'environnement "Maître"
    # Cet environnement sert de modèle pour créer les vecteurs d'environnements
    env = StandardCellEnv(
        cell_name=cell_name,
        pdk=pdk,
        max_steps=50,       # Un épisode = 50 essais max pour converger
        tolerance=0.05,     # Tolérance de 5% par rapport à la cible
        verbose=False,      # Silence sur l'env maître
        use_cache=True,      # Le cache accélère énormément le training !
        
    )
    
    # 3. Configuration de l'Agent PPO
    print(f"\n🤖 Initialisation de l'Agent PPO...")
    agent = RLAgent(
        env=env,
        weights_dir=Path("data/weight"), # Où sauvegarder les meilleurs poids JSON
        learning_rate=3e-4,              # Taux d'apprentissage standard PPO
        n_envs=n_envs,                   # Nombre de simulations en parallèle
        use_subprocess=(not force_cpu),  # True = Vrai multi-core (plus rapide mais + lourd)
        verbose=True,                     # Affiche les logs de training SB3
        max_no_improvement=10000
    )

    # 4. Lancement de l'entraînement
    print(f"\n🏃 Training en cours...")
    start_time = time.time()
    
    try:
        # save_freq = fréquence de sauvegarde des "meilleurs poids physiques" (JSON)
        best_cost = agent.train(total_timesteps=total_timesteps, save_freq=save_freq)
        
        duration = time.time() - start_time
        print(f"\n✅ Entraînement terminé en {duration:.1f}s ({duration/60:.1f} min)")
        print(f"🏆 Meilleur Coût Global atteint : {best_cost:.4f}")

    except KeyboardInterrupt:
        print("\n🛑 Entraînement interrompu par l'utilisateur.")
        print("💾 Sauvegarde du modèle actuel...")
        agent.model.save(f"data/models/{cell_name}_interrupted.zip")

    # 5. Validation rapide (Inference Test)
    print("\n🔍 Validation rapide sur une cible 'Performance'...")
    try:
        # On récupère la catégorie depuis l'environnement
        category = env.cell_category 
        json_path = Path("data/weight") / category / f"{cell_name}.json"
        
        if not json_path.exists():
            print(f"⚠️ Fichier de poids non trouvé : {json_path}")
            # On tente de charger le dernier modèle du callback si dispo
            return

        print(f"📂 Chargement depuis : {json_path}")
        best_widths = agent.weight_manager.load_weights(json_path)
        
        metrics = env.objective.evaluate(best_widths)
        print("metrics =", metrics  )
        print("-" * 40)
        print(f"Transistors : {best_widths}")
        print(f"Performance : Delay={metrics['delay_avg']:.2e}s | Power={metrics['energy_dyn']:.2e}J")
        print(f"Coût        : {metrics['cost']:.4f}")
        print("-" * 40)
        
    except Exception as e:
        print(f"⚠️ Impossible de charger les résultats finaux : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Gestion des arguments ligne de commande pour flexibilité
    parser = argparse.ArgumentParser(description="Train RL Agent for Standard Cell Optimization")
    parser.add_argument("--cell", type=str, default="sky130_fd_sc_hd__inv_1", help="Nom de la cellule")
    parser.add_argument("--steps", type=int, default=200, help="Nombre de steps total")
    parser.add_argument("--cores", type=int, default=4, help="Nombre de processus parallèles")
    parser.add_argument("--save_freq", type=int, default=12, help="Fréquence de sauvegarde des poids")
    
    args = parser.parse_args()
    
    train_agent(
        cell_name=args.cell,
        total_timesteps=args.steps,
        n_envs=args.cores,
        save_freq=args.save_freq
    )