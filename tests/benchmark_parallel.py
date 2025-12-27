# tests/test_parallel_rl.py
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from old.netlist_generator import SimulationConfig
from src.environment.gym_env import StandardCellEnv
from src.models.parallel_rl_agent import ParallelRLAgent
from src.models.weight_manager import WeightManager


def main():
    print("="*80)
    print("🚀 TEST RL PARALLÈLE")
    print("="*80)

    start_time = time.time()

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

    # 3. Environnement de BASE (un seul)
    env = StandardCellEnv(
        cell_name='sky130_fd_sc_hd__inv_1',
        pdk=pdk,
        config=config,
        cost_weights={
            'delay_avg': 0.5, 
            'delay_max': 0.6, 
            'tplh_avg': 0.4, 
            'tphl_avg': 0.4,
            'energy_dyn': 0.3, 
            'power_avg': 0.3, 
            'area': 0.2
        },
        max_steps=50,
        verbose=False,
        use_cache=True
    )

    # 4. ✅ Agent PARALLÈLE
    weights_dir = Path("./data/training_weights_parallel")
    
    agent = ParallelRLAgent(
        env,
        weights_dir=weights_dir,
        n_envs=12,              
        use_subprocess=True,   
        load_pretrained=False,
        
        # Hyperparamètres (optionnels)
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=10,
        verbose=True
    )

    # 5. Entraîner
    print("\n" + "="*80)
    print("🏋️  ENTRAÎNEMENT")
    print("="*80)
    
    best_cost = agent.train(
        total_timesteps=100,  # 20k steps distribués sur 8 envs
        save_freq=10
    )
    
    training_time = time.time() - start_time
    print(f"\n⏱️  Temps d'entraînement: {training_time:.1f}s")
    print(f"   Speedup estimé: ~{agent.n_envs}x vs séquentiel")

    # 6. Évaluer
    print("\n" + "="*80)
    print("📊 ÉVALUATION")
    print("="*80)
    
    mean_cost, std_cost, widths_history = agent.evaluate(n_episodes=10)
    
    print(f"\n📈 Résultats:")
    print(f"   Meilleur coût (train): {best_cost:.4f}")
    print(f"   Coût moyen (eval): {mean_cost:.4f} ± {std_cost:.4f}")
    print(f"   Amélioration: {(best_cost - mean_cost) / best_cost * 100:.1f}%")

    # 7. Vérifier les fichiers
    print("\n" + "="*80)
    print("📁 FICHIERS GÉNÉRÉS")
    print("="*80)
    
    weight_manager = WeightManager(base_dir=weights_dir)
    available_cells = weight_manager.list_available_cells()
    print(f"   Cellules disponibles: {available_cells}")
    
    # Exporter résumé
    summary_csv = weights_dir / "summary.csv"
    weight_manager.export_summary(summary_csv)
    print(f"   Résumé CSV: {summary_csv}")

    # 8. Cleanup
    agent.cleanup()
    
    print(f"\n✅ Test terminé!")
    print(f"   Temps total: {time.time() - start_time:.1f}s")
    print(f"   Poids sauvegardés: {weights_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
