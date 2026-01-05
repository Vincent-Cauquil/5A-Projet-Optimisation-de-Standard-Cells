import numpy as np
from stable_baselines3 import PPO
import sys
from pathlib import Path

# Ajout du chemin racine
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.environment.gym_env import StandardCellEnv
from src.simulation.pdk_manager import PDKManager

def query_agent(model_path, cell_name, target_constraints):
    """
    Demande à l'agent d'optimiser la cellule pour des contraintes précises.
    Retourne la meilleure configuration VALIDE trouvée.
    """
    print(f"\n🎯 OBJECTIF : {target_constraints}")
    
    pdk = PDKManager("sky130", verbose=False)
    
    # IMPORTANT : use_cache=False pour l'inférence afin de voir l'évolution réelle
    env = StandardCellEnv(
        cell_name, 
        pdk, 
        verbose=False, 
        use_cache=False,
        max_steps=20,
        tolerance=0.05  # Tolérance de 5% pour considérer la cible atteinte
        mode="inference"
    )
    
    # Chargement du modèle
    model = PPO.load(model_path)
    
    # Reset avec les cibles spécifiques
    obs, info = env.reset(options=target_constraints)
    
    print(f"   État initial : Delay={info['metrics']['delay_avg']:.2e}s, Power={info['metrics']['energy_dyn']:.2e}J")
    print(f"   Largeurs init: W_N={info['widths']['X0']*1e9:.0f}nm, W_P={info['widths']['X1']*1e9:.0f}nm")

    # Variables pour suivre le meilleur résultat valide
    best_reward = -float('inf')
    best_widths = info['widths']
    best_metrics = info['metrics']
    
    # Boucle d'optimisation (Inférence)
    for i in range(20): 
        # Prédiction déterministe (sans bruit aléatoire)
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        metrics = info['metrics']
        
        # Affichage du progrès
        delay_str = f"{metrics['delay_avg']:.2e}" if metrics['delay_avg'] != float('inf') else "INF"
        power_str = f"{metrics['energy_dyn']:.2e}" if metrics['energy_dyn'] != float('inf') else "INF"
        print(f"   Step {i+1:02d}: Delay={delay_str}s | P={power_str}J | Reward={reward:.2f}")
        
        # SAUVEGARDE DU MEILLEUR POINT VALIDE
        # On ne garde que si le reward est meilleur ET que la simulation n'a pas crashé (pas inf)
        if metrics['delay_avg'] != float('inf') and reward > best_reward:
            best_reward = reward
            best_widths = info['widths']
            best_metrics = metrics
        
        if terminated:
            print(f"   ✅ Cible atteinte (Tolérance respectée) !")
            break
            
    # On utilise les métriques du MEILLEUR step valide, pas forcément le dernier (qui peut être inf)
    final_metrics = best_metrics
    final_widths = best_widths
    
    print(f"   🛑 Fin Inférence. Meilleur Reward Valide: {best_reward:.2f}")
    
    # Vérification manuelle sur le meilleur résultat
    tgt_delay = target_constraints.get('delay_rise', float('inf'))
    delay_val = final_metrics['delay_rise']
    
    if delay_val != float('inf') and delay_val <= tgt_delay * 1.05: # Marge de 5%
        print(f"   🏆 SUCCÈS : Délai final {delay_val:.2e} <= Cible {tgt_delay:.2e}")
    else:
        print(f"   ⚠️  LIMITE : Délai final {delay_val:.2e} > Cible {tgt_delay:.2e}")

    return final_widths

if __name__ == "__main__":
    model_file = "data/models/inv/sky130_fd_sc_hd__inv_1.zip"
    cell_name = "sky130_fd_sc_hd__inv_1"

    if not Path(model_file).exists():
        print(f"❌ Modèle introuvable : {model_file}")
        exit()

    print("="*80)
    print("🤖 INTERROGATION DE L'AGENT (INFERENCE)")
    print("="*80)

    # --- SCÉNARIO 1 : Mode PERFORMANCE (Vitesse max) ---
    # On demande 30ps, ce qui est très agressif pour du 130nm
    w_fast = query_agent(model_file, cell_name, {
        "delay_rise": 30e-12, 
        "delay_fall": 30e-12,
        # On relâche les autres contraintes
        "power_dyn": 1e-3, 
        "area_um2": 10.0 
    })

    # --- SCÉNARIO 2 : Mode LOW POWER (Consommation min) ---
    # On accepte un délai lent (200ps) mais on veut une puissance minuscule
    w_eco = query_agent(model_file, cell_name, {
        "delay_rise": 200e-12,
        "delay_fall": 200e-12,
        # Cible puissance agressive
        "power_dyn": 1e-15, 
        "area_um2": 0.2 
    })
    
    print("\n" + "="*80)
    print("📊 COMPARAISON FINALE DES DESIGN")
    print("="*80)
    
    print(f"🚀 MODE PERFORMANCE :")
    print(f"   NMOS (X0) : {w_fast['X0']*1e9:.0f} nm")
    print(f"   PMOS (X1) : {w_fast['X1']*1e9:.0f} nm")
    print(f"   -> Doit être GROS pour aller vite.")

    print(f"\n🌱 MODE ÉCONOMIQUE :")
    print(f"   NMOS (X0) : {w_eco['X0']*1e9:.0f} nm")
    print(f"   PMOS (X1) : {w_eco['X1']*1e9:.0f} nm")
    print(f"   -> Doit être PETIT (proche de 420nm) pour moins consommer.")