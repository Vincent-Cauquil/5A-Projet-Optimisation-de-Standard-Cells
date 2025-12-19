# src/simulation/pool.py
import pandas as pd
from pathlib import Path
import subprocess
import tempfile
import re
from typing import List, Union
from .pdk_manager import PDKManager

class SequentialPool:
    """Pool séquentiel utilisant NGSpice en batch mode"""

    def __init__(self, netlist_paths: Union[str, List[str]], pdk_name: str = "sky130"):
        """
        Args:
            netlist_paths: Chemin vers une netlist ou liste de chemins
            pdk_name: Nom du PDK à utiliser (défaut: sky130)
        """
        # Initialiser le gestionnaire PDK
        self.pdk = PDKManager(pdk_name)
        
        # Normaliser en liste
        if isinstance(netlist_paths, str):
            netlist_paths = [netlist_paths]
        
        self.netlist_templates = []
        self.template_names = []
        
        for path in netlist_paths:
            netlist_path = Path(path)
            if not netlist_path.exists():
                raise FileNotFoundError(f"Netlist not found: {path}")
            
            with open(netlist_path, 'r') as f:
                content = f.read()
            
            # Remplacer automatiquement le chemin PDK
            content = self._inject_pdk_path(content)
            
            self.netlist_templates.append(content)
            self.template_names.append(netlist_path.stem)
    
    def _inject_pdk_path(self, netlist_content: str) -> str:
        """Remplace les chemins PDK par le chemin réel"""
        # Remplacer les patterns comme:
        # .lib /usr/local/share/pdk/sky130A/...
        # par le vrai chemin
        
        pattern = r'\.lib\s+[^\s]+sky130[^\s]*\.lib\.spice\s+(\w+)'
        
        def replacer(match):
            corner = match.group(1)  # tt, ss, ff, etc.
            return self.pdk.get_lib_include(corner)
        
        return re.sub(pattern, replacer, netlist_content)
    
    def run(self, params: pd.DataFrame) -> pd.DataFrame:
        """Exécute les simulations séquentiellement pour toutes les netlists"""
        all_results = []
        
        for idx, row in params.iterrows():
            print(f"\n{'='*60}")
            print(f"Simulation {idx+1}/{len(params)}")
            print(f"Paramètres: {row.to_dict()}")
            print('='*60)
            
            sim_results = {}
            
            # Exécuter chaque netlist avec les mêmes paramètres
            for template_content, template_name in zip(self.netlist_templates, self.template_names):
                print(f"\n▶ Netlist: {template_name}")
                
                try:
                    # Remplacer les paramètres
                    netlist_content = template_content
                    for param_name, param_value in row.items():
                        pattern = f'.param {param_name}=\\S+'
                        replacement = f'.param {param_name}={param_value}'
                        netlist_content = re.sub(pattern, replacement, netlist_content)
                    
                    # Créer fichier temporaire
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
                        f.write(netlist_content)
                        temp_file = f.name
                    
                    # Lancer ngspice
                    result = subprocess.run(
                        ['ngspice', '-b', temp_file],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    # Parser les mesures
                    measures = self._parse_measures(result.stdout)
                    
                    # Préfixer les mesures avec le nom de la netlist
                    for measure_name, measure_value in measures.items():
                        sim_results[f"{template_name}_{measure_name}"] = measure_value
                    
                    print(f"  ✓ Mesures: {measures}")
                    
                    # Nettoyer
                    Path(temp_file).unlink()
                    
                except subprocess.TimeoutExpired:
                    print(f"  ❌ TIMEOUT")
                    
                except Exception as e:
                    print(f"  ❌ Erreur: {e}")
            
            all_results.append(sim_results)
        
        return pd.DataFrame(all_results)
    
    def _parse_measures(self, stdout: str) -> dict:
        """Parse les mesures depuis la sortie NGSpice"""
        measures = {}
        for line in stdout.split('\n'):
            match = re.search(r'(\w+)\s*=\s*([0-9.e+-]+)', line)
            if match:
                measure_name = match.group(1)
                measure_value = float(match.group(2))
                measures[measure_name] = measure_value
        return measures
