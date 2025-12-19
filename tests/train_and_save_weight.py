# tests/test_rl_with_weight_manager.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import SimulationConfig
from src.environment.gym_env import StandardCellEnv
from src.models.rl_agent import RLAgent
from src.models.weight_manager import WeightManager

def main():
    print("="*80)
    print("🤖 TEST RL AVEC WEIGHTMANAGER")
    print("="*80)
    
    # 1. PDK
    pdk = PDKManager("sky130", verbose=True)
    
    # 2. Configuration
    config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner='tt',
        cload=10e-15,
        trise=100e-12,
        tfall=100e-12
    )
    
    # 3. Environnement
    env = StandardCellEnv(
        cell_name='sky130_fd_sc_hd__inv_1',
        pdk=pdk,
        config=config,
        cost_weights={'delay': 0.5, 'energy': 0.3, 'area': 0.2},
        max_steps=50,
        verbose=False,
        use_cache=True
    )
    
    # 4. Agent avec WeightManager
    weights_dir = Path("./data/training_weights")
    agent = RLAgent(
        env,
        weights_dir=weights_dir,
        load_pretrained=False  # Première fois: False
    )
    
    # 5. Entraîner
    print("\n🚀 Début de l'entraînement...")
    best_cost = agent.train(
        total_timesteps=10000,
        save_freq=500  
    )
    
    # 6. Évaluer
    print("\n📊 Évaluation finale...")
    mean_cost, std_cost, _ = agent.evaluate(n_episodes=10)
    
    # 7. Vérifier les fichiers sauvegardés
    print("\n📁 Fichiers générés:")
    weight_manager = WeightManager(base_dir=weights_dir)
    
    available_cells = weight_manager.list_available_cells()
    print(f"   Cellules disponibles: {available_cells}")
    
    # Exporter un résumé CSV
    summary_csv = weights_dir / "summary.csv"
    weight_manager.export_summary(summary_csv)
    
    print(f"\n✅ Test terminé avec succès!")
    print(f"   Meilleur coût: {best_cost:.4f}")
    print(f"   Poids sauvegardés: {weights_dir}")

if __name__ == "__main__":
    main()
