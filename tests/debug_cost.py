import sys
from pathlib import Path
import json

# Ajout du root au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import SimulationConfig
from src.optimization.objective import ObjectiveFunction

def debug_cost_calculation(cell_name="sky130_fd_sc_hd__inv_1"):
    print(f"\n🔍 --- DÉBOGAGE DU CALCUL DE COÛT : {cell_name} ---")
    
    # 1. Initialisation
    try:
        pdk = PDKManager("sky130", verbose=False)
        config = SimulationConfig() # Config par défaut
        
        # On active verbose=False pour ne pas spammer, on affichera nous-même les infos
        obj = ObjectiveFunction(
            cell_name=cell_name,
            config=config,
            pdk=pdk,
            verbose=True,
            use_cache=False # Important : on veut forcer le recalcul
        )
        
        print("✅ ObjectiveFunction initialisée.")
        
    except Exception as e:
        print(f"❌ Erreur critique à l'init : {e}")
        return

    # 2. Inspection de la Baseline chargée
    baseline_metrics = obj.baseline.get('metrics', {})
    print(f"\n📊 1. VÉRIFICATION DE LA BASELINE (Fichier JSON)")
    if not baseline_metrics:
        print("   ⚠️  AUCUNE BASELINE TROUVÉE ! Le coût sera de 1.0 ou erroné.")
        print("   👉 Solution : Lancez 'uv run python tests/generate_baselines.py'")
    else:
        print(f"   [Baseline] Delay Rise : {baseline_metrics.get('delay_rise', 'N/A')}")
        print(f"   [Baseline] Power Dyn  : {baseline_metrics.get('power_dyn', 'N/A')}")
        print(f"   [Baseline] Area (um²) : {baseline_metrics.get('area_um2', 'N/A')}")
        
        # Test rapide de cohérence
        area = baseline_metrics.get('area_um2', 0)
        if area > 1000:
            print("\n   🚨 ALERTE : L'aire de la baseline est GIGANTESQUE (> 1000 um²).")
            print("   🚨 C'est la preuve que votre baseline a été générée avec les mauvaises unités !")
            print("   👉 ACTION REQUISE : Supprimez 'src/models/references/*.json' et régénérez-les.")

    # 3. Simulation Actuelle (Tailles d'origine)
    print(f"\n⚡ 2. SIMULATION ACTUELLE (Tailles d'origine)")
    original_widths = obj.original_widths
    
    # Affichage des largeurs pour être sûr (en mètres)
    print(f"   Largeurs envoyées à la simu (Mètres) : {original_widths}")
    
    metrics = obj.evaluate(original_widths)
    print("metrics:", metrics)
    if metrics.get('cost') == obj.penalty_cost:
        print("❌ La simulation a échoué (Coût de pénalité). Vérifiez ngspice.")
        return

    print(f"   [Actuel]   Delay Rise : {metrics.get('delay_rise', 'N/A')}")
    print(f"   [Actuel]   Power Dyn  : {metrics.get('power_dyn', 'N/A')}")
    print(f"   [Actuel]   Area (um²) : {metrics.get('area_um2', 'N/A')}")

    # 4. Comparaison (Normalisation)
    print(f"\n⚖️  3. CALCUL DU COÛT (Ratios)")
    print(f"   Formule : Valeur Actuelle / Valeur Baseline (doit être proche de 1.0)")
    
    normalized = obj.get_normalized_metrics(cell_name, metrics)
    
    print(f"   {'Métrique':<20} | {'Ratio (Norm)':<15} | {'Verdict'}")
    print("   " + "-"*50)
    
    for key, ratio in normalized.items():
        if "cost" in key: continue
        
        verdict = "✅ OK"
        if ratio < 0.01: verdict = "⚠️ TROP PETIT (Baseline trop grande ?)"
        if ratio > 100:  verdict = "⚠️ TROP GRAND (Baseline trop petite ?)"
        
        print(f"   {key:<20} | {ratio:.6f}        | {verdict}")

    print(f"\n💰 COÛT FINAL CALCULÉ : {metrics.get('cost', 'N/A')}")

if __name__ == "__main__":
    debug_cost_calculation()