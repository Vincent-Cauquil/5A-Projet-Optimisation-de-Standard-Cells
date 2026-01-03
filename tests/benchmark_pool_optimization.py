# tests/benchmark_pool_optimization.py
#!/usr/bin/env python3
"""Benchmark simple des configurations de pool"""


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig
from src.simulation.pool import SequentialPool


def generate_test_netlists(n: int = 20) -> list:
    """Génère n netlists de test pour l'inverseur"""
    print(f"📝 Génération de {n} netlists...")
    
    pdk = PDKManager("sky130", verbose=False)
    config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner='tt',
        cload=10e-15,
        trise=200e-12,
        tfall=200e-12
    )
    
    generator = NetlistGenerator(pdk)
    
    sim_dir = Path("./benchmark_sims")
    sim_dir.mkdir(exist_ok=True)
    
    netlists = []
    cell_name = 'sky130_fd_sc_hd__inv_1'
    
    for i in range(n):
        output = sim_dir / f"test_inv_{i:04d}.cir"
        try:
            generator.generate_characterization_netlist(
                cell_name=cell_name,
                config=config,
                output_path=output
            )
            netlists.append(output)
        except Exception as e:
            print(f"⚠️  Erreur génération {i}: {e}")
    
    print(f"✅ {len(netlists)} netlists générées\n")
    return netlists


def benchmark_configurations():
    """Compare différentes configurations"""
    
    print("="*70)
    print("🚀 BENCHMARK POOL CONFIGURATIONS")
    print("="*70 + "\n")
    
    # Génération des tests
    n_sims = 20
    netlists = generate_test_netlists(n=n_sims)
    
    if not netlists:
        print("❌ Aucune netlist générée")
        return []
    
    pdk = PDKManager("sky130", verbose=False)
    config = SimulationConfig(
        vdd=1.8,
        temp=27,
        corner='tt',
        cload=10e-15
    )
    
    configs = [
        {'name': 'Default', 'fast_mode': False, 'env_opt': False},
        {'name': 'Fast Mode', 'fast_mode': True, 'env_opt': False},
        {'name': 'Env Optimized', 'fast_mode': False, 'env_opt': True},
        {'name': 'Full Optimized', 'fast_mode': True, 'env_opt': True},
    ]
    
    results = []
    
    for idx, cfg in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"🔬 Test {idx}/4: {cfg['name']}")
        print(f"   fast_mode={cfg['fast_mode']}, env_optimized={cfg['env_opt']}")
        print("="*70)
        
        try:
            pool = SequentialPool(
                pdk, config,
                fast_mode=cfg['fast_mode'],
                env_optimized=cfg['env_opt'],
                verbose=False
            )
            
            start = time.time()
            df = pool.run_batch(netlists)
            duration = time.time() - start
            
            # ✅ CORRECTION : Comptage correct des succès
            n_success = len(df) - df['error'].notna().sum()  # Lignes sans erreur
            n_total = len(df)
            avg_time = (duration / n_total * 1000) if n_total > 0 else 0
            
            results.append({
                'config': cfg['name'],
                'time': duration,
                'speedup': results[0]['time'] / duration if results else 1.0,
                'success': n_success,
                'total': n_total
            })
            
            print(f"⏱️  Temps total: {duration:.2f}s")
            print(f"📊 Temps moyen: {avg_time:.2f}ms/sim")
            print(f"✅ Succès: {n_success}/{n_total} ({100*n_success/n_total:.1f}%)")
            
        except KeyboardInterrupt:
            print("\n❌ Test interrompu par l'utilisateur")
            raise
        except Exception as e:
            print(f"❌ ERREUR durant le test: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'config': cfg['name'],
                'time': None,
                'speedup': None,
                'success': 0,
                'total': 0
            })
    
    # Affichage résumé
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES RÉSULTATS")
    print("="*70)
    print(f"{'Configuration':<20} {'Temps':<10} {'Speedup':<10} {'Succès':<15}")
    print("-"*70)
    
    for r in results:
        if r['time'] is not None:
            print(f"{r['config']:<20} {r['time']:>6.2f}s    {r['speedup']:>5.2f}x      "
                  f"{r['success']}/{r['total']:<10}")
        else:
            print(f"{r['config']:<20} {'FAILED':<10} {'-':<10} {r['success']}/{r['total']:<10}")
    
    # Estimation
    if results and results[0]['time']:
        print(f"\n💡 Estimation pour 100 simulations:")
        for r in results:
            if r['time']:
                est_time = r['time'] * (100 / n_sims)
                print(f"   {r['config']:<20} ~{est_time:.1f}s (~{est_time/60:.1f}min)")
    
    print("="*70 + "\n")
    return results


def quick_test():
    """Test rapide avec seulement 2 configurations"""
    print("="*50)
    print("⚡ QUICK TEST (5 simulations)")
    print("="*50 + "\n")
    
    netlists = generate_test_netlists(n=5)
    
    if not netlists:
        print("❌ Aucune netlist")
        return
    
    pdk = PDKManager("sky130", verbose=False)
    config = SimulationConfig(vdd=1.8, temp=27)
    
    # Test 1: Default
    print("1️⃣  Default...")
    pool1 = SequentialPool(pdk, config, fast_mode=False, env_optimized=False)
    t1 = time.time()
    df1 = pool1.run_batch(netlists)
    time1 = time.time() - t1
    n_success1 = len(df1) - df1['error'].notna().sum()
    print(f"   ⏱️  {time1:.2f}s | ✅ {n_success1}/5\n")
    
    # Test 2: Full Optimized
    print("2️⃣  Full Optimized...")
    pool2 = SequentialPool(pdk, config, fast_mode=True, env_optimized=True)
    t2 = time.time()
    df2 = pool2.run_batch(netlists)
    time2 = time.time() - t2
    n_success2 = len(df2) - df2['error'].notna().sum()
    speedup = time1 / time2 if time2 > 0 else 0
    print(f"   ⏱️  {time2:.2f}s | ✅ {n_success2}/5")
    print(f"   🚀 Speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark pool configurations')
    parser.add_argument('--quick', action='store_true', 
                       help='Test rapide (5 sims au lieu de 20)')
    parser.add_argument('-n', '--num-sims', type=int, default=20,
                       help='Nombre de simulations (défaut: 20)')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            quick_test()
        else:
            benchmark_configurations()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrompu\n")
