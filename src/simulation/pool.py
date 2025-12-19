# src/simulation/pool.py
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import re

class SequentialPool:
    """Pool séquentiel utilisant NGSpice en batch mode"""
    
    def __init__(self, netlist_path: str):
        self.netlist_template = Path(netlist_path)
        
        if not self.netlist_template.exists():
            raise FileNotFoundError(f"Netlist not found: {netlist_path}")
        
        # Lire le template
        with open(self.netlist_template, 'r') as f:
            self.template_content = f.read()
    
    def run(self, params: pd.DataFrame) -> pd.DataFrame:
        """Exécute les simulations séquentiellement"""
        results = []
        
        for idx, row in params.iterrows():
            print(f"\n{'='*60}")
            print(f"Simulation {idx+1}/{len(params)}")
            print(f"Paramètres: {row.to_dict()}")
            print('='*60)
            
            try:
                # Remplacer les paramètres dans la netlist
                netlist_content = self.template_content
                
                for param_name, param_value in row.items():
                    # Remplacer .param R_val=1k par .param R_val=<valeur>
                    pattern = f'.param {param_name}=\\S+'
                    replacement = f'.param {param_name}={param_value}'
                    netlist_content = re.sub(pattern, replacement, netlist_content)
                
                print("Contenu netlist modifiée:")
                print(netlist_content[:200])  # Debug: afficher début
                
                # Créer fichier temporaire
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
                    f.write(netlist_content)
                    temp_file = f.name
                
                print(f"Fichier temporaire: {temp_file}")
                
                # Lancer ngspice en batch mode avec timeout
                print("Lancement NGSpice...")
                result = subprocess.run(
                    ['ngspice', '-b', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10  # Timeout de 10 secondes
                )
                
                print("✓ Simulation terminée")
                
                # Afficher sortie pour debug
                if result.returncode != 0:
                    print(f"stderr: {result.stderr[:500]}")
                
                # Parser les mesures
                measures = {}
                for line in result.stdout.split('\n'):
                    # Chercher lignes du type: fc = 1.591549e+02
                    match = re.search(r'(\w+)\s*=\s*([0-9.e+-]+)', line)
                    if match:
                        measure_name = match.group(1)
                        measure_value = float(match.group(2))
                        measures[measure_name] = measure_value
                
                print(f"Mesures extraites: {measures}")
                results.append(measures)
                
                # Nettoyer
                Path(temp_file).unlink()
                
            except subprocess.TimeoutExpired:
                print("❌ TIMEOUT - simulation bloquée après 10s")
                results.append({})
                
            except Exception as e:
                print(f"❌ Erreur: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                results.append({})
        
        return pd.DataFrame(results)