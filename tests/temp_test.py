import sys
import json
from pathlib import Path

# Ajout du root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.gym_env import StandardCellEnv
from src.simulation.pdk_manager import PDKManager

def test_cost_logic():
    print("="*80)
    print("🔬 TEST DE VALIDATION DU CALCUL DE COÛT (RL REWARD)")
    print("="*80)

    cell_name = "sky130_fd_sc_hd__inv_1"
    pdk = PDKManager("sky130", verbose=False)
    
    print(f"\n1️⃣  Initialisation de l'environnement pour {cell_name}...")
    # On force use_cache=False pour être sûr de recalculer
    env = StandardCellEnv(cell_name, pdk, verbose=True, use_cache=False)
    
    print(f"\n2️⃣  Vérification de la Baseline...")
    # On récupère les métriques brutes de la baseline
    baseline_data = env.objective.baseline # C'est le dictionnaire complet du JSON
    ref_metrics = baseline_data.get('metrics', {})
    print(f"   ℹ️  Baseline Data Keys: {list(baseline_data.keys())}")
    if not ref_metrics:
        print("❌ ERREUR: Aucune métrique dans la baseline !")
        return

    print(f"   ✅ Baseline trouvée.")
    print(f"   ℹ️  Références (clés disponibles : {list(ref_metrics.keys())}) :")
    # CORRECTION ICI : On utilise les clés réelles du JSON (energy_dyn)
    print(f"      - Delay Avg Ref : {ref_metrics.get('delay_avg', 0):.2e} s")
    print(f"      - Energy Dyn Ref: {ref_metrics.get('energy_dyn', 0):.2e} J") 
    print(f"      - Area Ref      : {ref_metrics.get('area_um2', 0):.2f} um²")

    print(f"\n3️⃣  Configuration des poids de coût...")
    # On définit les poids correspondants aux clés normalisées
    correct_weights = {
        'delay_avg_norm': 0.5,
        'energy_dyn_norm': 0.3,
        'area_um2_norm': 0.2
    }

    print(f"\n4️⃣  Lancement de env.reset()...")
    obs, info = env.reset()
    metrics = info['metrics']

    print(f"\n5️⃣  Analyse des résultats :")
    print(f"   {'Métrique':<20} | {'Mesure':<12} | {'Baseline':<12} | {'Ratio':<8}")
    print("-" * 60)
    
    # Mapping entre nom de métrique runtime et nom de métrique baseline
    # Runtime (Objective) -> Baseline (JSON)
    key_mapping = {
        'delay_avg': 'delay_avg',
        'energy_dyn': 'energy_dyn', # Match direct maintenant
        'area_um2': 'area_um2'
    }
    
    calculated_cost = 0.0
    total_weight = 0.0
    
    for key, ref_key in key_mapping.items():
        val = metrics.get(key, 0)
        ref = ref_metrics.get(ref_key, 1e-9) 
        
        ratio = val / ref
        print(f"   {key:<20} | {val:.2e} | {ref:.2e} | {ratio:.4f}")
        
        norm_key = f"{key}_norm"
        if norm_key in correct_weights:
            w = correct_weights[norm_key]
            calculated_cost += ratio * w
            total_weight += w

    manual_cost = calculated_cost / total_weight if total_weight > 0 else 1000.0
    env_cost = metrics.get('cost', -1)
    
    print(f"\n   [B] Validation du Coût :")
    print(f"      - Coût Env    : {env_cost:.6f}")
    print(f"      - Coût Manuel : {manual_cost:.6f}")
    
    if abs(env_cost - manual_cost) < 0.001:
         print(f"   ✅ SUCCÈS : Cohérence parfaite.")
    else:
         print(f"   ⚠️  DISCRÉPANCE : Vérifiez les poids dans ObjectiveFunction.")

if __name__ == "__main__":
    test_cost_logic()
