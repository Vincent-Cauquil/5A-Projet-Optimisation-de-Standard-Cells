import sys
import numpy as np
from pathlib import Path

# Ajout du root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.gym_env import StandardCellEnv
from src.simulation.pdk_manager import PDKManager

def test_variation_impact():
    print("="*80)
    print("📉 TEST DE VARIATION PHYSIQUE (ACTION -> IMPACT)")
    print("="*80)

    cell_name = "sky130_fd_sc_hd__inv_1"
    pdk = PDKManager("sky130", verbose=False)
    
    # 1. Init
    # On force use_cache=False pour être sûr de simuler la nouvelle netlist
    env = StandardCellEnv(cell_name, pdk, verbose=False, use_cache=False)
    
    # Poids pour le test (identiques à ceux par défaut corrigés)
    weights = {
        'delay_avg_norm': 0.5,
        'energy_dyn_norm': 0.3,
        'area_um2_norm': 0.2
    }
    
    # 2. Reset (Baseline)
    print(f"\n1️⃣  État Initial (Reset)...")
    obs, info = env.reset()
    base_widths = env.current_widths.copy()
    print(f"   Largeurs Base : {base_widths}")
    print(f"   Coût Base     : {info['metrics']['cost']:.4f}")

    # 3. Action : Augmenter W de +20% (+0.2)
    # Pour un inverseur, il y a 2 transistors (N et P)
    # L'action est un tableau numpy de deltas
    action = np.array([0.2, 0.2], dtype=np.float32)
    
    print(f"\n2️⃣  Application Action : {action} (+20% sur tous les W)")
    
    # 4. Step
    obs, reward, terminated, truncated, info = env.step(action)
    
    new_metrics = info['metrics']
    new_widths = env.current_widths
    
    # 5. Analyse
    print(f"\n3️⃣  Analyse de l'impact :")
    
    # A. Largeurs
    print(f"\n   [A] Largeurs (nm) :")
    print(f"   {'Transistor':<10} | {'Avant':<10} | {'Après':<10} | {'Delta':<8}")
    print("-" * 50)
    for name in base_widths:
        old_w = base_widths[name] * 1e9
        new_w = new_widths[name] * 1e9
        delta = (new_w - old_w) / old_w * 100
        print(f"   {name:<10} | {old_w:.0f}       | {new_w:.0f}       | {delta:+.1f}%")

    # B. Métriques & Coût
    ref_metrics = env.objective.baseline['metrics']
    
    print(f"\n   [B] Métriques & Coût :")
    print(f"   {'Métrique':<20} | {'Baseline':<10} | {'Nouveau':<10} | {'Ratio':<8} | {'Poids':<6}")
    print("-" * 75)
    
    mapping = {
        'delay_avg': ('delay_avg_norm', 's'),
        'energy_dyn': ('energy_dyn_norm', 'J'), # energy_dyn est utilisé par objective
        'area_um2': ('area_um2_norm', 'um²')
    }
    
    calc_cost = 0.0
    tot_weight = 0.0
    
    for key, (norm_key, unit) in mapping.items():
        base = ref_metrics.get(key, 1e-9)
        curr = new_metrics.get(key, 0)
        
        ratio = curr / base
        weight = weights.get(norm_key, 0.0)
        
        calc_cost += ratio * weight
        tot_weight += weight
        
        print(f"   {key:<20} | {base:.2e} | {curr:.2e} | {ratio:.4f}   | {weight:.1f}")

    manual_final_cost = calc_cost / tot_weight
    env_final_cost = new_metrics['cost']
    
    print(f"\n   [C] Résultat Final :")
    print(f"      - Coût Calculé : {manual_final_cost:.6f}")
    print(f"      - Coût Env     : {env_final_cost:.6f}")
    
    # Validation Logique Physique
    print(f"\n   [D] Vérification Physique :")
    is_faster = new_metrics['delay_avg'] < ref_metrics['delay_avg']
    is_bigger = new_metrics['area_um2'] > ref_metrics['area_um2']
    is_hungry = new_metrics['energy_dyn'] > ref_metrics['energy_dyn']
    
    print(f"      - Plus rapide ? {'✅ OUI' if is_faster else '❌ NON'} (Attendu : OUI car W augmente)")
    print(f"      - Plus gros ?   {'✅ OUI' if is_bigger else '❌ NON'} (Attendu : OUI car W augmente)")
    print(f"      - Plus gourmand?{'✅ OUI' if is_hungry else '❌ NON'} (Attendu : OUI car Cload interne augmente)")

if __name__ == "__main__":
    test_variation_impact()