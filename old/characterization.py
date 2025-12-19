# src/simulation/characterization.py
import pandas as pd
import numpy as np
from pathlib import Path
from ..src.simulation.pool import SequentialPool
from ..src.simulation.pdk_manager import PDKManager

class XOR2Characterization:
    """Caractérisation complète de XOR2 avec auto-détection PDK"""
    
    def __init__(self, netlist_dir: str = "netlists/templates/xor2"):
        self.netlist_dir = Path(netlist_dir)
        
        # Initialiser PDK et Pool
        self.pdk = PDKManager("sky130")
        
        self.netlists = {
            'delay': self.netlist_dir / "delay.cir",
            'static': self.netlist_dir / "static.cir",
            'energy': self.netlist_dir / "energy.cir",
            'dynamic': self.netlist_dir / "dynamic.cir"
        }
        
        # Vérifier que toutes les netlists existent
        for name, path in self.netlists.items():
            if not path.exists():
                raise FileNotFoundError(f"Netlist {name} not found: {path}")
    
    def run_full_characterization(self, param_sweep: pd.DataFrame) -> pd.DataFrame:
        """
        Caractérisation complète avec plusieurs jeux de paramètres
        
        Args:
            param_sweep: DataFrame avec colonnes SUPPLY, TEMP, CLOAD, FREQ, trise, tfall
            
        Returns:
            DataFrame avec tous les résultats
        """
        print(f"\n{'='*60}")
        print(f"Caractérisation XOR2 - {len(param_sweep)} configurations")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for netlist_name, netlist_path in self.netlists.items():
            print(f"\n🔹 Netlist: {netlist_name}")
            
            # Créer un pool pour cette netlist
            pool = SequentialPool(str(netlist_path), pdk_name="sky130")
            
            # Filtrer les paramètres pertinents
            if netlist_name == 'static':
                params = param_sweep[['SUPPLY', 'TEMP']].copy()
            else:
                params = param_sweep.copy()
            
            # Exécuter les simulations
            try:
                results = pool.run(params)
                
                # Préfixer les colonnes avec le nom de la netlist
                results = results.add_prefix(f"{netlist_name}_")
                
                all_results.append(results)
                
            except Exception as e:
                print(f"❌ Erreur {netlist_name}: {e}")
        
        # Fusionner tous les résultats
        if all_results:
            final_results = pd.concat([param_sweep] + all_results, axis=1)
            
            # Calculer des métriques supplémentaires
            final_results = self._compute_metrics(final_results)
            
            return final_results
        
        return param_sweep
    
    def _compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule des métriques dérivées"""
        
        # Délai moyen (si les colonnes existent)
        delay_cols = [c for c in df.columns if c.startswith('delay_') and 'delay_avg' not in c]
        if delay_cols:
            df['delay_avg'] = df[delay_cols].mean(axis=1)
        
        # Consommation statique moyenne
        static_cols = [c for c in df.columns if c.startswith('static_L_') and 'avg' not in c]
        if static_cols:
            df['static_L_avg'] = df[static_cols].mean(axis=1)
        
        # PDP (Power-Delay Product)
        if 'delay_avg' in df.columns and 'dynamic_Pdynamic' in df.columns:
            df['PDP'] = df['delay_avg'] * df['dynamic_Pdynamic']
        
        # EDP (Energy-Delay Product)
        if 'delay_avg' in df.columns and 'energy_Etot_avg' in df.columns:
            df['EDP'] = df['delay_avg'] * df['energy_Etot_avg']
        
        return df
    
    def save_results(self, results: pd.DataFrame, output_file: str = "results/xor2_characterization.csv"):
        """Sauvegarde les résultats"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results.to_csv(output_path, index=False)
        print(f"\n✓ Résultats sauvegardés: {output_path}")
        
        return output_path
