#!/usr/bin/env python3
"""
Caractérisation XOR2 avec PyNGSpice
"""

from pathlib import Path
from pyngs.core import NGSpiceInstance
import pandas as pd
import numpy as np

class XOR2Characterization:
    """Classe pour caractériser la cellule XOR2"""
    
    def __init__(self, netlist_dir="src/model"):
        self.netlist_dir = Path(netlist_dir)
        self.netlists = {
            'delay': self.netlist_dir / "delay_xor2.spice",
            'static': self.netlist_dir / "static_power_xor2.spice",
            'energy': self.netlist_dir / "energy_xor2.spice",
            'dynamic': self.netlist_dir / "dynamic_power_xor2.spice"
        }
    
    def run_delay_characterization(self, params):
        """
        Caractérisation du délai
        
        Args:
            params: dict avec SUPPLY, TEMP, CLOAD, trise, tfall
        """
        print(f"\n[Délai] Simulation avec {params}")
        
        inst = NGSpiceInstance()
        inst.load(str(self.netlists['delay']))
        
        # Configurer les paramètres
        for param, value in params.items():
            inst.set_parameter(param, value)
        
        # Exécuter la simulation
        inst.run()
        
        # Extraire les mesures
        results = {
            'delay_B_rise_A0': inst.get_measure('delay_B_rise_A0'),
            'delay_A_rise_B1': inst.get_measure('delay_A_rise_B1'),
            'delay_B_fall_A1': inst.get_measure('delay_B_fall_A1'),
            'delay_A_fall_B0': inst.get_measure('delay_A_fall_B0'),
        }
        
        # Calculer le délai moyen
        results['delay_avg'] = np.mean(list(results.values()))
        
        inst.stop()
        
        return results
    
    def run_static_power_characterization(self, params):
        """
        Caractérisation de la consommation statique
        
        Args:
            params: dict avec SUPPLY, TEMP
        """
        print(f"\n[Statique] Simulation avec {params}")
        
        inst = NGSpiceInstance()
        inst.load(str(self.netlists['static']))
        
        # Configurer les paramètres
        for param, value in params.items():
            inst.set_parameter(param, value)
        
        # Exécuter la simulation
        inst.run()
        
        # Extraire les mesures pour chaque état
        results = {
            'L_00': inst.get_measure('L_00'),
            'L_01': inst.get_measure('L_01'),
            'L_10': inst.get_measure('L_10'),
            'L_11': inst.get_measure('L_11'),
        }
        
        # Calculer la consommation moyenne
        results['L_avg'] = np.mean(list(results.values()))
        
        inst.stop()
        
        return results
    
    def run_energy_characterization(self, params):
        """
        Caractérisation de l'énergie
        
        Args:
            params: dict avec SUPPLY, TEMP, FREQ, CLOAD
        """
        print(f"\n[Énergie] Simulation avec {params}")
        
        inst = NGSpiceInstance()
        inst.load(str(self.netlists['energy']))
        
        # Configurer les paramètres
        for param, value in params.items():
            inst.set_parameter(param, value)
        
        # Exécuter la simulation
        inst.run()
        
        # Extraire les mesures
        results = {
            'Etot_avg': inst.get_measure('Etot_avg'),
            'Iavg_total': inst.get_measure('Iavg_total'),
            'Pavg_total': inst.get_measure('Pavg_total'),
        }
        
        inst.stop()
        
        return results
    
    def run_dynamic_power_characterization(self, params):
        """
        Caractérisation de la consommation dynamique
        
        Args:
            params: dict avec SUPPLY, TEMP, FREQ, CLOAD
        """
        print(f"\n[Dynamique] Simulation avec {params}")
        
        inst = NGSpiceInstance()
        inst.load(str(self.netlists['dynamic']))
        
        # Configurer les paramètres
        for param, value in params.items():
            inst.set_parameter(param, value)
        
        # Exécuter la simulation
        inst.run()
        
        # Extraire les mesures
        results = {
            'Pstatic': inst.get_measure('Pstatic'),
            'Pdynamic': inst.get_measure('Pdynamic'),
            'Energy_per_switch': inst.get_measure('Energy_per_switch'),
        }
        
        inst.stop()
        
        return results
    
    def run_full_characterization(self, param_sweep):
        """
        Caractérisation complète avec plusieurs jeux de paramètres
        
        Args:
            param_sweep: DataFrame pandas avec les paramètres
        """
        all_results = []
        
        for idx, params in param_sweep.iterrows():
            print(f"\n{'='*60}")
            print(f"Configuration {idx+1}/{len(param_sweep)}")
            print(f"{'='*60}")
            
            result = params.to_dict()
            
            # Délai
            try:
                delay_results = self.run_delay_characterization(params)
                result.update(delay_results)
            except Exception as e:
                print(f"[ERREUR Délai] {e}")
            
            # Consommation statique
            try:
                static_params = {k: v for k, v in params.items() if k in ['SUPPLY', 'TEMP']}
                static_results = self.run_static_power_characterization(static_params)
                result.update(static_results)
            except Exception as e:
                print(f"[ERREUR Statique] {e}")
            
            # Énergie
            try:
                energy_results = self.run_energy_characterization(params)
                result.update(energy_results)
            except Exception as e:
                print(f"[ERREUR Énergie] {e}")
            
            # Consommation dynamique
            try:
                dynamic_results = self.run_dynamic_power_characterization(params)
                result.update(dynamic_results)
            except Exception as e:
                print(f"[ERREUR Dynamique] {e}")
            
            all_results.append(result)
        
        return pd.DataFrame(all_results)


def main():
    """Fonction principale"""
    
    print("="*60)
    print("Caractérisation XOR2_1 - Sky130 PDK avec PyNGSpice")
    print("="*60)
    
    # Créer l'objet de caractérisation
    char = XOR2Characterization()
    
    # Définir les paramètres de sweep
    param_sweep = pd.DataFrame({
        "SUPPLY": [1.8, 1.8, 1.8, 1.65, 1.95],
        "TEMP": [27, -40, 125, 27, 27],
        "CLOAD": [10e-15, 10e-15, 10e-15, 50e-15, 10e-15],
        "FREQ": [100e6, 100e6, 100e6, 100e6, 50e6],
        "trise": [100e-12, 100e-12, 100e-12, 100e-12, 500e-12],
        "tfall": [100e-12, 100e-12, 100e-12, 100e-12, 500e-12]
    })
    
    print("\n[Paramètres de sweep]")
    print(param_sweep)
    
    # Exécuter la caractérisation complète
    results = char.run_full_characterization(param_sweep)
    
    # Sauvegarder les résultats
    output_file = "results/xor2_characterization.csv"
    Path("results").mkdir(exist_ok=True)
    results.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"[Résultats sauvegardés] {output_file}")
    print(f"{'='*60}")
    
    # Afficher un résumé
    print("\n[RÉSUMÉ DES RÉSULTATS]")
    print(results)
    
    # Statistiques
    numeric_cols = results.select_dtypes(include=[np.number]).columns
    print("\n[STATISTIQUES]")
    print(results[numeric_cols].describe())


if __name__ == "__main__":
    main()
