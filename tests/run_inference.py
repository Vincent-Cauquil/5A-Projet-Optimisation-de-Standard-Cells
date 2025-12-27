import numpy as np
from stable_baselines3 import PPO
import sys
from pathlib import Path

# Ajout du chemin racine
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.environment.gym_env import StandardCellEnv
from src.simulation.pdk_manager import PDKManager

def query_agent(model, cell_name, target_constraints):
    """
    Demande à l'agent d'optimiser la cellule pour des contraintes précises.
    """
    print(f"\n🎯 OBJECTIF : {target_constraints}")
    
    pdk = PDKManager("sky130", verbose=False)
    # On passe les options ici pour FORCER la cible
    env = StandardCellEnv(cell_name, pdk, verbose=False)
    
    # Reset avec vos cibles spécifiques
    obs, info = env.reset(options=target_constraints)
    
    print(f"   État initial : Delay={info['metrics']['delay_avg']:.2e}s, Power={info['metrics']['energy_dyn']:.2e}J")

    done = False
    total_reward = 0
    
    # Boucle d'optimisation
    for i in range(20): # L'agent a 20 coups pour converger
        # On utilise 'model' passé en argument
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"   ✅ Cible atteinte au step {i+1} !")
            break
            
    final_metrics = info['metrics']
    print(f"   RÉSULTAT     : Delay={final_metrics['delay_avg']:.2e}s, Power={final_metrics['energy_dyn']:.2e}J")
    
    # Vérification
    tgt_delay = target_constraints.get('delay_rise', float('inf'))
    
    # On vérifie si on est proche de la cible (tolérance ou <=)
    if final_metrics['delay_rise'] <= tgt_delay:
        print("   🏆 Succès : Contrainte de délai respectée.")
    else:
        print("   ⚠️  Échec : Délai trop grand (Limites physiques atteintes ?)")

    return info['widths']

if __name__ == "__main__":
    # 1. Charger le modèle entraîné
    model_path = "data/models/inv/sky130_fd_sc_hd__inv_1.zip"
    if not Path(model_path).exists():
        print("❌ Modèle non trouvé. Lancez train.py d'abord.")
        exit()
        
    print(f"📂 Chargement du modèle : {model_path}")
    model = PPO.load(model_path)
    cell = "sky130_fd_sc_hd__inv_1"

    print("="*80)
    print("🤖 INTERROGATION DE L'AGENT (INFERENCE)")
    print("="*80)

    # SCÉNARIO 1 : Performance Max (Délai très court)
    # On demande 30ps (très rapide)
    w_fast = query_agent(model, cell, {
        "delay_rise": 30e-12, 
        "delay_fall": 30e-12,
        "power_dyn": 1.0,    # On s'en fiche (valeur haute)
        "area_um2": 10.0     # On s'en fiche
    })

    # SCÉNARIO 2 : Low Power (Consommation minimale)
    # On demande un délai relaxé (200ps) mais une puissance très faible
    w_low_power = query_agent(model, cell, {
        "delay_rise": 200e-12,
        "delay_fall": 200e-12,
        "power_dyn": 1e-15,  # Très faible cible
        "area_um2": 0.5      # Petite surface
    })
    
    print("\n📊 COMPARAISON DES CHOIX DE L'IA :")
    print(f"   Mode Rapide    : W_N={w_fast['X0']*1e9:.0f}nm, W_P={w_fast['X1']*1e9:.0f}nm")
    print(f"   Mode Éco       : W_N={w_low_power['X0']*1e9:.0f}nm, W_P={w_low_power['X1']*1e9:.0f}nm")