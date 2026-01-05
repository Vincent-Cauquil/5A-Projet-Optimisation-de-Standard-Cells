#!/usr/bin/env python3
# src/simulation/spice_runner.py
# ============================================================
#  Spice Runner
# ============================================================
"""
Exécuteur de simulations NGSpice avec support multi-moteur.
Gère l'exécution via binaire CLI ou via bibliothèque partagée (pyngs).

Auteurs : Vincent Cauquil (vincent.cauquil@cpe.fr)
          Léonard Anselme (leonard.anselme@cpe.fr)

Date : Novembre 2025 - Janvier 2026

class SpiceRunner :
    run_simulation : Point d'entrée pour lancer une simulation et parser les résultats.
    _run_with_cli : Exécution rapide via l'exécutable ngspice système.
    _run_with_lib : Exécution via l'API C de NGSpice (si disponible).
    _parse_cli_output : Extraction des valeurs .meas depuis la sortie texte.
"""
import subprocess
import shutil
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from pyngs.core import NGSpiceInstance
    HAS_PYNGS = True
except ImportError:
    HAS_PYNGS = False

class SpiceRunner:
    def __init__(self, pdk_root: Path, mode: str = "cli", verbose: bool = False):
        """
        Args:
            mode: 'auto' (préfère lib, fallback sur cli), 'lib' (force lib), 'cli' (force exe)
        """
        self.pdk_root = pdk_root
        self.verbose = verbose
        self.mode = mode
        
        # Détection de l'exécutable système (pour le mode CLI)
        self.cli_path = shutil.which("ngspice")
        
        # Logique de décision
        if self.mode == "lib" and not HAS_PYNGS:
            raise ImportError("Mode 'lib' demandé mais pyngs n'est pas installé.")
        
        if self.mode == "cli" and not self.cli_path:
            raise FileNotFoundError("Mode 'cli' demandé mais 'ngspice' introuvable (sudo apt install ngspice).")

    def run_simulation(self, netlist_path: Union[str, Path], verbose: bool = False) -> Dict:
        """Point d'entrée principal : choisit la meilleure méthode"""
        path_obj = Path(netlist_path).resolve()
        
        # Décision du moteur
        use_cli = False
        if self.mode == "cli":
            use_cli = True
        elif self.mode == "lib":
            use_cli = False
        elif self.mode == "auto":
            # Par défaut, on préfère le CLI s'il est là (souvent plus rapide/stable)
            use_cli = (self.cli_path is not None)
            
        # Exécution
        if use_cli and self.cli_path:
            return self._run_with_cli(path_obj, verbose)
        elif HAS_PYNGS:
            return self._run_with_lib(path_obj, verbose)
        else:
            return {
                'success': False, 
                'measures': {}, 
                'errors': ["Aucun moteur NGSpice disponible (ni 'ngspice' système, ni 'pyngs')."]
            }

    # ==========================================
    # MÉTHODE 1 : LIBRAIRIE (Lent)
    # ==========================================
    def _run_with_lib(self, netlist_path: Path, verbose: bool) -> Dict:
        expected_keys = self._scan_meas_names(netlist_path)
        inst = NGSpiceInstance()
        
        try:
            inst.load(str(netlist_path))
            inst.run()
            
            raw_measures = {}
            for key in expected_keys:
                try:
                    raw_measures[key] = inst.get_measure(key)
                except Exception:
                    raw_measures[key] = None
            
            final_measures = self._post_process_measures(raw_measures, netlist_path)
            
            return {
                'success': True,
                'measures': final_measures,
                'errors': [],
                'mode': 'lib'
            }
        except Exception as e:
            return {'success': False, 'measures': {}, 'errors': [str(e)], 'mode': 'lib'}
        finally:
            inst.stop()

    # ==========================================
    # MÉTHODE 2 : CLI  (Rapide)
    # ==========================================
    def _run_with_cli(self, netlist_path: Path, verbose: bool) -> Dict:
        if not self.cli_path:
             return {'success': False, 'measures': {}, 'errors': ["ngspice executable not found"], 'mode': 'cli'}

        cmd = [self.cli_path, "-b", str(netlist_path)]
        
        try:
            # Lancement du processus
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                env={**os.environ, 'NGSPICE_MEMINIT': '0'} # Optimisation mineure
            )
            
            # Parsing du texte (stdout)
            raw_measures = self._parse_cli_output(result.stdout)
            
            # Vérification des erreurs critiques dans stderr/stdout
            errors = []
            if result.returncode != 0:
                errors.append(f"Exit code {result.returncode}")
                # Ajouter ici ta fonction _check_errors si tu veux filtrer stderr
            
            final_measures = self._post_process_measures(raw_measures, netlist_path)
            
            return {
                'success': result.returncode == 0 and len(final_measures) > 0,
                'measures': final_measures,
                'errors': errors,
                'mode': 'cli'
            }
            
        except Exception as e:
            return {'success': False, 'measures': {}, 'errors': [str(e)], 'mode': 'cli'}

    # ==========================================
    # UTILITAIRES COMMUNS
    # ==========================================
    def _scan_meas_names(self, netlist_path: Path) -> List[str]:
        """Pour le mode LIB : Trouve ce qu'il faut demander"""
        try:
            text = netlist_path.read_text(encoding='utf-8')
            keys = re.findall(r'^\s*\.meas\s+(?:ac|tran|dc)\s+(\w+)', text, re.MULTILINE | re.IGNORECASE)
            return [k.lower() for k in keys]
        except Exception:
            return []

    def _parse_cli_output(self, stdout: str) -> Dict[str, float]:
        """Pour le mode CLI : Lit le texte de sortie"""
        measures = {}
        # Pattern standard NGSpice : nom_variable = 1.2345e-09
        pattern = r'^([a-zA-Z0-9_]+)\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)'
        
        for line in stdout.split('\n'):
            line = line.strip()
            if 'failed' in line.lower(): continue
            
            match = re.match(pattern, line)
            if match:
                name = match.group(1).lower()
                try:
                    measures[name] = float(match.group(2))
                except ValueError:
                    pass
        return measures

    def _post_process_measures(self, measures: Dict[str, float], netlist_path: Path) -> Dict[str, float]:
        """
        Ta logique métier existante (Moyennes, Power, etc.)
        Copiée-collée de ton ancien code ou importée.
        """
        # --- Insère ici ta logique de calcul de moyennes ---
        # Exemple simple pour que le code tourne :
        if 'energy_dyn' in measures and measures['energy_dyn'] is not None:
             measures['power_avg'] = measures['energy_dyn'] / 2e-9 # Exemple
             
        return measures