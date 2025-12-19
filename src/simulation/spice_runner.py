# src/simulation/spice_runner.py
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd

class SpiceRunner:
    """Exécute des simulations NGSpice et extrait les résultats"""

    def __init__(self, pdk_root: Path):
        self.pdk_root = pdk_root
        self.ngspice_dir = pdk_root / "libs.tech" / "ngspice"

    def run_simulation(
        self, 
        netlist_path: Path, 
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
                cwd=str(self.ngspice_dir),  # ← Clé : bon répertoire de travail
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

            # Extraire les mesures
            measures = self._extract_measurements(result.stdout)

            # Vérifier les erreurs
            errors = self._check_errors(result.stdout, result.stderr)

            success = (result.returncode == 0 and len(errors) == 0)

            if verbose:
                if errors:
                    print("\n⚠️  ERREURS DÉTECTÉES:")
                    for error in errors:
                        print(f"   • {error}")
                else:
                    print("\n✅ Simulation terminée sans erreur")

                if measures:
                    print(f"\n📊 Mesures extraites: {len(measures)}")
                    for key, value in measures.items():
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

        # Pattern pour les mesures
        # Format: nom = valeur (avec unité optionnelle)
        pattern = r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'

        for line in stdout.split('\n'):
            line = line.strip()
            match = re.match(pattern, line)
            if match:
                name = match.group(1).lower()
                value_str = match.group(2)

                # Ignorer les lignes de debug/info
                skip_names = {
                    'temp', 'available', 'size', 'pages', 'stack',
                    'warning', 'note', 'index', 'total'
                }
                if name in skip_names:
                    continue

                try:
                    value = float(value_str)
                    measures[name] = value
                except ValueError:
                    continue

        return measures

    def _check_errors(self, stdout: str, stderr: str) -> List[str]:
        """Détecte les erreurs dans la sortie"""
        errors = []

        # Patterns d'erreurs communes
        error_patterns = [
            r'Error:',
            r'Fatal error',
            r'analysis not run',
            r'convergence',
            r'singular matrix',
            r'timestep too small',
            r'undefined',
            r'failed',
        ]

        combined = stdout + '\n' + stderr

        for pattern in error_patterns:
            matches = re.finditer(pattern, combined, re.IGNORECASE)
            for match in matches:
                # Extraire la ligne complète
                start = combined.rfind('\n', 0, match.start()) + 1
                end = combined.find('\n', match.end())
                if end == -1:
                    end = len(combined)
                error_line = combined[start:end].strip()
                
                # Filtrer les notes/warnings non critiques
                if error_line and error_line not in errors:
                    if not error_line.lower().startswith(('note:', 'warning: total')):
                        errors.append(error_line)

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
