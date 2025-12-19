# src/simulation/spice_runner.py
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, Optional, List, Union
import pandas as pd
import os

class SpiceRunner:
    """Exécute des simulations NGSpice et extrait les résultats"""

    def __init__(self, pdk_root: Path, worker_id: Optional[int] = None):
        self.pdk_root = pdk_root
        self.ngspice_dir = pdk_root / "libs.tech" / "ngspice"
        self.worker_id = worker_id or os.getpid()

    def run_simulation(
        self, 
        netlist_path: Union[str, Path], 
        verbose: bool = False
    ) -> Dict:
        """
        Exécute une simulation NGSpice

        Args:
            netlist_path: Chemin vers la netlist
            verbose: Afficher la sortie complète

        Returns:
            Dict avec success, measures, errors, stdout, stderr
        """
        # Convertir en chemin absolu
        netlist_abs = netlist_path.absolute()

        if not netlist_abs.exists():
            return {
                'success': False,
                'measures': {},
                'errors': [f"Netlist introuvable: {netlist_abs}"],
                'stdout': '',
                'stderr': ''
            }

        if verbose:
            print(f"\n🔧 Simulation: {netlist_path.name}")
            print("="*60)

        # Commande NGSpice
        cmd = ["ngspice", "-b", str(netlist_abs)]

        # IMPORTANT: Exécuter depuis le répertoire ngspice du PDK
        # pour que les includes relatifs fonctionnent
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.ngspice_dir),
                capture_output=True,
                text=True,
                timeout=60
            )

            if verbose:
                print("📤 STDOUT:")
                print(result.stdout)
                if result.stderr:
                    print("\n📤 STDERR:")
                    print(result.stderr)

            # Extraire les mesures brutes
            measures = self._extract_measurements(result.stdout)

            # ✅ POST-TRAITEMENT: Calculer les métriques dérivées
            measures = self._post_process_measures(measures, netlist_abs)

            # Vérifier les erreurs CRITIQUES uniquement
            errors = self._check_errors(result.stdout, result.stderr)

            success = (result.returncode == 0 and len(errors) == 0 and len(measures) > 0)

            if verbose:
                if errors:
                    print("\n⚠️  ERREURS CRITIQUES:")
                    for error in errors:
                        print(f"   • {error}")
                else:
                    print("\n✅ Simulation terminée sans erreur")

                if measures:
                    print(f"\n📊 Mesures finales: {len(measures)}")
                    for key, value in sorted(measures.items()):
                        print(f"   • {key}: {value}")
                else:
                    print("\n⚠️  Aucune mesure extraite")

            return {
                'success': success,
                'measures': measures,
                'errors': errors,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            error_msg = "Simulation timeout (>60s)"
            if verbose:
                print(f"❌ {error_msg}")
            return {
                'success': False,
                'measures': {},
                'errors': [error_msg],
                'stdout': '',
                'stderr': ''
            }
        except Exception as e:
            error_msg = f"Erreur d'exécution: {str(e)}"
            if verbose:
                print(f"❌ {error_msg}")
            return {
                'success': False,
                'measures': {},
                'errors': [error_msg],
                'stdout': '',
                'stderr': ''
            }

    def _extract_measurements(self, stdout: str) -> Dict[str, float]:
        """
        Extrait les mesures .meas de la sortie NGSpice

        Format NGSpice:
        delay_a_rise_b0         =  1.234567e-11
        """
        measures = {}

        # Pattern pour les mesures (première occurrence uniquement)
        pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'

        for line in stdout.split('\n'):
            line = line.strip()
            
            # ✅ Ignorer les lignes avec "failed"
            if 'failed' in line.lower():
                continue
            
            match = re.match(pattern, line)
            if match:
                name = match.group(1).lower()
                value_str = match.group(2)

                # Ignorer les lignes de debug/info
                skip_names = {
                    'temp', 'available', 'size', 'pages', 'stack',
                    'warning', 'note', 'index', 'total', 'tnom',
                    'reference', 'rows'
                }
                if name in skip_names:
                    continue

                try:
                    value = float(value_str)
                    # Garder uniquement la première occurrence
                    if name not in measures:
                        measures[name] = value
                except ValueError:
                    continue

        return measures

    def _post_process_measures(self, measures: Dict[str, float], 
                               netlist_path: Path) -> Dict[str, float]:
        """
        Post-traite les mesures pour calculer des métriques dérivées
        
        Calcule:
        - power_avg depuis energy_dyn
        - tplh_avg/tphl_avg depuis les délais individuels
        - delay_avg (moyenne des montées et descentes)
        """
        # Calculer power_avg depuis energy_dyn
        if 'energy_dyn' in measures:
            total_time = self._extract_simulation_time(netlist_path)
            if total_time and total_time > 0:
                measures['power_avg'] = measures['energy_dyn'] / total_time
        
        # Calculer les délais moyens
        tplh_values = [v for k, v in measures.items() if k.startswith('tplh_t')]
        tphl_values = [v for k, v in measures.items() if k.startswith('tphl_t')]
        
        if tplh_values:
            measures['tplh_avg'] = sum(tplh_values) / len(tplh_values)
        
        if tphl_values:
            measures['tphl_avg'] = sum(tphl_values) / len(tphl_values)
        
        # Délai moyen global
        if 'tplh_avg' in measures and 'tphl_avg' in measures:
            measures['delay_avg'] = (measures['tplh_avg'] + measures['tphl_avg']) / 2
        
        return measures

    def _extract_simulation_time(self, netlist_path: Path) -> Optional[float]:
        """
        Extrait le temps total de simulation depuis la netlist
        
        Cherche la ligne: .tran <step> <stop_time>
        Formats supportés: 12n, 1.5u, 100p, 1e-9, etc.
        """
        try:
            with open(netlist_path, 'r') as f:
                content = f.read()
            
            # Pattern: .tran <step> <stop>
            match = re.search(r'\.tran\s+\S+\s+(\S+)', content, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                
                # Parser les suffixes d'unité
                if time_str.endswith('n'):
                    return float(time_str[:-1]) * 1e-9
                elif time_str.endswith('u'):
                    return float(time_str[:-1]) * 1e-6
                elif time_str.endswith('p'):
                    return float(time_str[:-1]) * 1e-12
                elif time_str.endswith('m'):
                    return float(time_str[:-1]) * 1e-3
                else:
                    return float(time_str)
                    
        except Exception:
            pass
        
        return None

    def _check_errors(self, stdout: str, stderr: str) -> List[str]:
        """
        Détecte UNIQUEMENT les erreurs critiques
        
        Ignore les warnings bénins comme:
        - "insertnumber: fails" (comportement normal de NGSpice)
        - "vector XXX is not available" (re-run interne)
        - "= failed" isolé (mesure échouée en re-run)
        """
        errors = []

        critical_patterns = [
            r'Fatal error',
            r'analysis not run due to errors',
            r'singular matrix',
            r'timestep too small',
            r'Error on line \d+',
            r'undefined element',
            r'convergence failed',
            r'too few input',
            r'unrecognized',
        ]

        ignore_patterns = [
            r'insertnumber: fails',           # NGSpice interne
            r'vector .* is not available',    # Re-run normal
            r'=\s*failed\s*$',                # Mesure failed en re-run
            r'Note:',                          # Notes informatives
            r'Warning: total',                # Stats mémoire
            r'Reference value',               # Debug info
            r'No\. of Data Rows',             # Stats simulation
        ]

        combined = stdout + '\n' + stderr

        for line in combined.split('\n'):
            line = line.strip()
            if not line:
                continue

            # Ignorer les warnings bénins
            if any(re.search(p, line, re.IGNORECASE) for p in ignore_patterns):
                continue

            # Chercher les erreurs critiques
            for pattern in critical_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if line not in errors:
                        errors.append(line)
                    break

        return errors

    def run_batch(
        self, 
        netlist_files: List[Path], 
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Exécute plusieurs simulations

        Returns:
            DataFrame avec toutes les mesures
        """
        results = []

        for netlist_file in netlist_files:
            result = self.run_simulation(netlist_file, verbose)

            if result['success'] and result['measures']:
                row = {'netlist': netlist_file.name}
                row.update(result['measures'])
                results.append(row)

        return pd.DataFrame(results)
