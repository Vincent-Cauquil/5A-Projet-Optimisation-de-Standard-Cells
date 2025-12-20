# tests/test_rl_simple.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import SimulationConfig
from src.environment.gym_env import StandardCellEnv
from src.models.rl_agent import RLAgent
import time

def main():
    print("🚀 TEST SIMPLE RL")
    
    # Setup
    pdk = PDKManager("sky130", verbose=False)
    config = SimulationConfig(vdd=1.8, temp=27, corner='tt', cload=10e-15, trise=100e-12, tfall=100e-12)
    
    env = StandardCellEnv(
        cell_name='sky130_fd_sc_hd__inv_1',
        pdk=pdk,
        config=config,
        cost_weights={'delay_avg': 0.5, 'energy_dyn': 0.3, 'area': 0.2},
        verbose=False,
        use_cache=True
    )

    agent = RLAgent(env, weights_dir=Path("./data/weights"), 
                    load_pretrained=False, n_envs=8)

    # ✅ ENTRAÎNEMENT ULTRA-COURT
    print("\n⏳ Entraînement 20 steps max...")
    start = time.time()
    
    try:
        best_cost = agent.train(total_timesteps=1000, save_freq=100)
        elapsed = time.time() - start
        
        print(f"\n✅ FINI en {elapsed:.1f}s")
        print(f"   Meilleur coût: {best_cost:.4f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrompu (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")

if __name__ == "__main__":
    main()
