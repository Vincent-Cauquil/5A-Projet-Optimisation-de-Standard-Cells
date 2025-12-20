# tests/quick_bench_save.py
from pathlib import Path
import sys
import time
import shutil
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import SimulationConfig
from src.environment.gym_env import StandardCellEnv
from src.models.parallel_rl_agent import ParallelRLAgent
from src.models.weight_manager import WeightManager

def main():
    print("="*80)
    print("⚡ BENCH RAPIDE - TEST SAUVEGARDE")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Nettoyage précédent (optionnel)
    weights_dir = Path("./data/quick_bench_weights")
    if weights_dir.exists():
        print(f"\n🗑️  Nettoyage: {weights_dir}")
        shutil.rmtree(weights_dir)
    
    # 2. PDK + Config
    pdk = PDKManager("sky130", verbose=False)
    config = SimulationConfig(
        vdd=1.8, temp=27, corner='tt',
        cload=10e-15, trise=100e-12, tfall=100e-12
    )
    
    # 3. Environnement
    cell_name = 'sky130_fd_sc_hd__inv_1'
    env = StandardCellEnv(
        cell_name=cell_name,
        pdk=pdk,
        config=config,
        cost_weights={'delay_avg': 0.5, 'energy_dyn': 0.3, 'area': 0.2},
        max_steps=20,  # ✅ Réduit pour aller vite
        verbose=False,
        use_cache=True
    )
    
    # 4. Agent parallèle (peu d'envs pour test rapide)
    print(f"\n🤖 Création agent parallèle...")
    agent = ParallelRLAgent(
        env,
        weights_dir=weights_dir,
        n_envs=4,  # ✅ Seulement 4 pour test rapide
        use_subprocess=True,
        load_pretrained=False,
        verbose=False
    )
    
    # 5. Mini entraînement avec saves fréquents
    print(f"\n🏋️  Entraînement minimal (200 steps)...")
    print(f"   → 4 envs × ~50 steps/env")
    print(f"   → Sauvegarde tous les 50 steps\n")
    
    best_cost = agent.train(
        total_timesteps=200,
        save_freq=50,  # ✅ Save fréquent pour tester
        log_interval=1,
        progress_bar=True
    )
    
    train_time = time.time() - start_time
    
    # 6. Vérification des fichiers
    print("\n" + "="*80)
    print("📁 VÉRIFICATION DES SAUVEGARDES")
    print("="*80)
    
    weight_manager = WeightManager(base_dir=weights_dir)
    
    # Lister les fichiers créés
    cell_dir = weights_dir / cell_name
    if cell_dir.exists():
        files = sorted(cell_dir.glob("*.zip"))
        print(f"✅ Dossier cellule: {cell_dir}")
        print(f"   Nombre de checkpoints: {len(files)}")
        
        for f in files:
            size_kb = f.stat().st_size / 1024
            print(f"   - {f.name} ({size_kb:.1f} KB)")
        
        # Vérifier best_model.zip
        best_model = cell_dir / "best_model.zip"
        if best_model.exists():
            print(f"\n✅ Meilleur modèle trouvé: {best_model.name}")
            print(f"   Taille: {best_model.stat().st_size / 1024:.1f} KB")
        else:
            print(f"\n⚠️  Pas de best_model.zip trouvé")
    else:
        print(f"❌ Dossier cellule non créé: {cell_dir}")
    
    # Tester le chargement
    print("\n" + "="*80)
    print("🔄 TEST RECHARGEMENT")
    print("="*80)
    
    try:
        agent_reload = ParallelRLAgent(
            env,
            weights_dir=weights_dir,
            n_envs=4,
            load_pretrained=True,  # ✅ Recharger
            verbose=False
        )
        print(f"✅ Rechargement réussi!")
        
        # Évaluation rapide
        mean_cost, std_cost, _ = agent_reload.evaluate(n_episodes=3, verbose=False)
        print(f"   Coût moyen: {mean_cost:.4f} ± {std_cost:.4f}")
        
        agent_reload.cleanup()
        
    except Exception as e:
        print(f"❌ Erreur rechargement: {e}")
    
    # 7. Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ")
    print("="*80)
    
    print(f"⏱️  Temps total: {time.time() - start_time:.1f}s")
    print(f"   - Entraînement: {train_time:.1f}s")
    print(f"🎯 Meilleur coût: {best_cost:.4f}")
    print(f"📁 Poids: {weights_dir}")
    
    # Export CSV
    summary_csv = weights_dir / "summary.csv"
    weight_manager.export_summary(summary_csv)
    print(f"📄 Résumé CSV: {summary_csv}")
    
    # Cleanup
    agent.cleanup()
    
    print(f"\n✅ Bench terminé!\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Interruption")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
