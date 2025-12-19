# tests/test_rl_parallel.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import SimulationConfig
from src.environment.vectorized_env import VectorizedStandardCellEnv
from src.models.rl_agent import RLAgent
from src.models.weight_manager import WeightManager
import time

def main():
    print("="*80)
    print("⚡ TEST RL PARALLÈLE AVEC VECTORIZED ENVS")
    print("="*80)

    start_time = time.time()

    # 1. PDK
    pdk = PDKManager("sky130", verbose=True)

    # 2. Configuration RAPIDE
    config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner='tt',
        cload=10e-15,
        trise=200e-12,
        tfall=200e-12
    )

    # 3. ✅ Environnement VECTORISÉ (4 envs en parallèle)
    vec_env = VectorizedStandardCellEnv(
        cell_name='sky130_fd_sc_hd__inv_1',
        pdk=pdk,
        config=config,
        cost_weights={'delay': 0.5, 'energy': 0.3, 'area': 0.2},
        max_steps=20,
        n_envs=4,          # ✅ 4 environnements parallèles
        use_cache=True
    )

    # 4. Agent avec env vectorisé
    weights_dir = Path("./data/training_weights_parallel")
    agent = RLAgent(
        vec_env,  # ✅ Passer le VectorizedEnv
        weights_dir=weights_dir,
        load_pretrained=False
    )

    # 5. Entraînement PARALLÈLE
    print("\n🚀 Début de l'entraînement PARALLÈLE...")
    print(f"   - 4 environnements en parallèle")
    print(f"   - 5000 timesteps")
    
    best_cost = agent.train(
        total_timesteps=5000,  # Avec 4 envs = 20000 simulations effectives
        save_freq=500
    )

    training_time = time.time() - start_time

    # 6. Évaluation (sur un env simple précis)
    print("\n📊 Évaluation finale (mode précis)...")
    
    from src.environment.gym_env import StandardCellEnv
    
    eval_config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner='tt',
        cload=10e-15,
        trise=100e-12,
        tfall=100e-12
    )
    
    eval_env = StandardCellEnv(
        cell_name='sky130_fd_sc_hd__inv_1',
        pdk=pdk,
        config=eval_config,
        cost_weights={'delay': 0.5, 'energy': 0.3, 'area': 0.2},
        max_steps=50,
        verbose=False,
        use_cache=False,
    )
    
    # Évaluer sur l'env précis
    from stable_baselines3 import PPO
    model = PPO.load(str(weights_dir / "sky130_fd_sc_hd__inv_1_best.zip"))
    
    costs = []
    for _ in range(5):
        obs, _ = eval_env.reset()
        done = False
        episode_cost = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_cost = info.get('cost', 0)
        costs.append(episode_cost)
    
    mean_cost = np.mean(costs)
    std_cost = np.std(costs)

    # 7. Résultats
    print(f"\n✅ Test terminé avec succès!")
    print(f"   Temps d'entraînement: {training_time:.1f}s ({training_time/60:.1f}min)")
    print(f"   Meilleur coût (training): {best_cost:.4f}")
    print(f"   Coût moyen (eval précise): {mean_cost:.4f} ± {std_cost:.4f}")
    print(f"   Poids sauvegardés: {weights_dir}")
    
    # Speedup estimé
    baseline_time = 180  # 3 minutes version séquentielle
    speedup = baseline_time / training_time
    print(f"\n⚡ Speedup: {speedup:.1f}x par rapport à la version séquentielle")

if __name__ == "__main__":
    import numpy as np
    main()
