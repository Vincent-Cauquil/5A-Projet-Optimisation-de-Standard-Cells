# src/simulation/pool.py
import pandas as pd
from pathlib import Path
from pyngs.core import NGSpiceInstance

class SequentialPool:
    """Exécution séquentielle de simulations SPICE"""
    
    def __init__(self, netlist_path: Path):
        self.netlist_path = netlist_path
    
    def run(self, params_df: pd.DataFrame) -> pd.DataFrame:
        """
        Exécute les simulations pour chaque ligne de params_df
        
        Args:
            params_df: DataFrame avec colonnes = paramètres SPICE
        
        Returns:
            DataFrame avec les mesures extraites
        """
        results = []
        
        for idx, row in params_df.iterrows():
            inst = NGSpiceInstance()
            inst.load(str(self.netlist_path))
            
            # Configurer tous les paramètres
            for param_name, param_value in row.items():
                inst.set_parameter(param_name, param_value)
            
            # Simuler
            inst.run()
            
            # Extraire toutes les mesures
            measures = {}
            for measure_name in inst.list_measures():
                measures[measure_name] = inst.get_measure(measure_name)
            
            results.append(measures)
            inst.stop()
        
        return pd.DataFrame(results)
