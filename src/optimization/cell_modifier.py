# src/optimization/cell_modifier.py

"""
Module de modification des paramètres de transistors dans les netlists SPICE.

Supporte:
- Transistors MOSFET directs (M0, M1, ...)
- Instances de sous-circuits (X0, X1, ...)
- Notation scientifique (w=1e+06u)
- Unités SPICE (u, m, n, p)

Auteur: Projet I4-COMSC
PDK: Sky130
"""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class CellModifier:
    """
    Modifie les paramètres W/L des transistors dans une netlist SPICE.
    
    Exemples d'utilisation:
        >>> modifier = CellModifier("/tmp/inv.sp")
        >>> modifier.modify_width('X0', 700.0)  # Nouvelle largeur en nm
        >>> modifier.apply_modifications("/tmp/inv_modified.sp")
    """
    
    def __init__(self, netlist_path: str):
        """
        Initialise le modificateur avec une netlist SPICE.
        
        Args:
            netlist_path: Chemin vers la netlist SPICE
            
        Raises:
            FileNotFoundError: Si la netlist n'existe pas
            ValueError: Si aucun transistor modifiable n'est trouvé
        """
        self.netlist_path = Path(netlist_path)
        
        if not self.netlist_path.exists():
            raise FileNotFoundError(f"Netlist introuvable: {netlist_path}")
        
        # Charger le contenu
        with open(self.netlist_path, 'r') as f:
            self.content = f.read()
        
        # Extraire les transistors
        self.transistors = self._extract_transistors()
        
        if not self.transistors:
            raise ValueError(
                f"Aucun transistor modifiable trouvé dans {netlist_path}\n"
                "Vérifiez que la netlist contient des lignes du type:\n"
                "  X0 ... sky130_fd_pr__nfet_01v8 w=XXXu l=YYYu"
            )
    
    
    def _extract_transistors(self) -> Dict[str, Dict[str, float]]:
        """
        Extrait les transistors et leurs paramètres de la netlist.
        
        Cherche les patterns:
        - M0 ... sky130_fd_pr__nfet_01v8 w=650000u l=150000u
        - X1 ... sky130_fd_pr__pfet_01v8_hvt w=1e+06u l=150000u
        
        Returns:
            Dict de la forme:
            {
                'X0': {'w': 650.0, 'l': 150.0, 'type': 'nfet', 'model': '...'},
                'X1': {'w': 1000.0, 'l': 150.0, 'type': 'pfet', 'model': '...'},
            }
            Les valeurs w et l sont en nanomètres.
        """
        transistors = {}
        
        # Pattern regex amélioré pour capturer tous les formats
        # Groupe 1: Nom (M0, X1, ...)
        # Groupe 2: Modèle (sky130_fd_pr__nfet_01v8)
        # Groupes 3-4: Largeur + unité (650000, 'u')
        # Groupes 5-6: Longueur + unité (150000, 'u')
        pattern = (
            r'^([MX]\d+)\s+'           # Nom du transistor
            r'.*?\s+'                   # Noeuds (ignore)
            r'(sky130_fd_pr__[np]fet\S*)\s+'  # Modèle
            r'.*?'                      # Autres paramètres (ignore)
            r'w=([\d.eE+-]+)([umpn]?)'  # w=VALUE + unité
            r'.*?'                      # Séparateur
            r'l=([\d.eE+-]+)([umpn]?)'  # l=VALUE + unité
        )
        
        for match in re.finditer(pattern, self.content, re.MULTILINE | re.IGNORECASE):
            name = match.group(1)              # 'X0', 'M1', etc.
            model = match.group(2)             # 'sky130_fd_pr__nfet_01v8'
            w_val = float(match.group(3))     # 650000.0 ou 1e+06
            w_unit = match.group(4) or 'u'    # 'u', 'n', 'm', 'p'
            l_val = float(match.group(5))     # 150000.0
            l_unit = match.group(6) or 'u'    # 'u', 'n', 'm', 'p'
            
            # Convertir en nanomètres
            w_nm = self._to_nanometers(w_val, w_unit)
            l_nm = self._to_nanometers(l_val, l_unit)
            
            # Déterminer le type (nfet ou pfet)
            trans_type = 'nfet' if 'nfet' in model.lower() else 'pfet'
            
            transistors[name] = {
                'w': w_nm,
                'l': l_nm,
                'type': trans_type,
                'model': model
            }
        
        return transistors
    
    
    def _to_nanometers(self, value: float, unit: str) -> float:
        """
        Convertit une valeur SPICE Sky130 en nanomètres "logiques".
        
        ⚠️ CONVENTION SKY130:
        Dans les fichiers SPICE Sky130, les largeurs sont données en "sous-unités":
        - w=650000u signifie 650000 unités de base
        - 1 unité de base = 1nm dans le PDK
        - Donc w=650000u = 650000nm de base = 650µm réels
        
        Pour simplifier la manipulation, on normalise:
        - w=650000u → 650 nm "logiques" (division par 1000)
        - w=1e+06u → 1000 nm "logiques"
        
        Args:
            value: Valeur numérique brute (ex: 650000, 1e+06)
            unit: Unité SPICE ('u', 'm', 'n', 'p')
        
        Returns:
            Valeur en nanomètres "logiques" pour manipulation
        
        Examples:
            >>> _to_nanometers(650000, 'u')
            650.0  # nm logiques
            
            >>> _to_nanometers(1e6, 'u')
            1000.0  # nm logiques
            
            >>> _to_nanometers(150000, 'u')
            150.0  # nm logiques
        """
        # Dans Sky130: 1u = 1nm de base
        # On divise par 1000 pour obtenir des µm "logiques" = nm "logiques"
        
        conversions = {
            'u': 0.001,      # 650000u → 650 nm logiques
            'n': 1.0,        # 150n → 150 nm
            'm': 1e6,        # 1m → 1e6 nm (cas théorique)
            'p': 0.001,      # 1p → 0.001 nm (cas théorique)
            '': 0.001        # Par défaut: comme 'u'
        }
        
        return value * conversions.get(unit.lower(), 0.001)


    def _from_nanometers(self, value_nm: float, target_unit: str = 'u') -> Tuple[float, str]:
        """
        Convertit des nanomètres "logiques" vers unités SPICE Sky130.
        
        Conversion inverse de _to_nanometers().
        
        Args:
            value_nm: Valeur en nm logiques (ex: 700, 1200)
            target_unit: Unité cible (toujours 'u' pour Sky130)
        
        Returns:
            (valeur en unités Sky130, unité)
        
        Examples:
            >>> _from_nanometers(650, 'u')
            (650000.0, 'u')  # Redevient w=650000u
            
            >>> _from_nanometers(700, 'u')
            (700000.0, 'u')  # Devient w=700000u
            
            >>> _from_nanometers(1200, 'u')
            (1200000.0, 'u')  # Devient w=1200000u
        """
        # Conversion inverse: nm logiques → unités Sky130
        # 700 nm logiques → 700000u dans SPICE
        
        conversions = {
            'u': 1000.0,     # 700 nm → 700000u
            'n': 1.0,        # 700 nm → 700n
            'm': 1e-6,       # 700 nm → 0.0000007m
            'p': 1000.0,     # 700 nm → 700000p
            '': 1000.0       # Par défaut: comme 'u'
        }
        
        if target_unit not in conversions:
            target_unit = 'u'
        
        return value_nm * conversions[target_unit], target_unit
    
    
    def get_transistor_widths(self) -> Dict[str, float]:
        """
        Retourne les largeurs actuelles de tous les transistors.
        
        Returns:
            Dict {'X0': 650.0, 'X1': 1000.0, ...} en nanomètres
        """
        return {name: params['w'] for name, params in self.transistors.items()}
    
    
    def get_transistor_info(self, transistor_name: str) -> Dict[str, float]:
        """
        Retourne toutes les informations d'un transistor.
        
        Args:
            transistor_name: Nom du transistor (ex: 'X0', 'M1')
        
        Returns:
            Dict avec clés: 'w', 'l', 'type', 'model'
        
        Raises:
            KeyError: Si le transistor n'existe pas
        """
        if transistor_name not in self.transistors:
            available = ', '.join(self.transistors.keys())
            raise KeyError(
                f"Transistor '{transistor_name}' introuvable.\n"
                f"Transistors disponibles: {available}"
            )
        return self.transistors[transistor_name].copy()
    
    
    def modify_width(self, transistor_name: str, new_width_nm: float) -> None:
        """
        Modifie la largeur d'un transistor (en mémoire uniquement).
        
        La modification n'est pas appliquée au fichier tant que
        apply_modifications() n'est pas appelée.
        
        Args:
            transistor_name: Nom du transistor (ex: 'X0', 'M1')
            new_width_nm: Nouvelle largeur en nanomètres
        
        Raises:
            KeyError: Si le transistor n'existe pas
            ValueError: Si la largeur est invalide (<= 0)
        
        Examples:
            >>> modifier.modify_width('X0', 700.0)  # 700nm
            >>> modifier.modify_width('X1', 1.2e3)  # 1200nm
        """
        if transistor_name not in self.transistors:
            available = ', '.join(self.transistors.keys())
            raise KeyError(
                f"Transistor '{transistor_name}' introuvable.\n"
                f"Transistors disponibles: {available}"
            )
        
        if new_width_nm <= 0:
            raise ValueError(
                f"Largeur invalide: {new_width_nm}nm (doit être > 0)"
            )
        
        # Contraintes du PDK Sky130
        MIN_WIDTH = 150.0   # 150nm minimum
        MAX_WIDTH = 10000.0 # 10µm maximum raisonnable
        
        if not (MIN_WIDTH <= new_width_nm <= MAX_WIDTH):
            import warnings
            warnings.warn(
                f"Largeur {new_width_nm}nm hors limites recommandées "
                f"[{MIN_WIDTH}, {MAX_WIDTH}]nm. Clipping appliqué."
            )
            new_width_nm = np.clip(new_width_nm, MIN_WIDTH, MAX_WIDTH)
        
        self.transistors[transistor_name]['w'] = float(new_width_nm)
    
    
    def modify_multiple_widths(self, widths: Dict[str, float]) -> None:
        """
        Modifie plusieurs largeurs en une seule fois.
        
        Args:
            widths: Dict {'X0': 700.0, 'X1': 1200.0, ...} en nanomètres
        
        Examples:
            >>> modifier.modify_multiple_widths({
            ...     'X0': 700.0,
            ...     'X1': 1200.0
            ... })
        """
        for name, width in widths.items():
            self.modify_width(name, width)
    
    
    def apply_modifications(self, output_path: Optional[str] = None) -> str:
        """
        Applique les modifications et sauvegarde la netlist.
        
        Remplace les valeurs w=XXXu par les nouvelles valeurs calculées.
        
        Args:
            output_path: Chemin de sortie (None = écrase l'original)
        
        Returns:
            Chemin du fichier modifié
        
        Examples:
            >>> output = modifier.apply_modifications("/tmp/inv_modified.sp")
        """
        modified_content = self.content
        
        # Remplacer chaque transistor modifié
        for name, params in self.transistors.items():
            # Pattern pour trouver la ligne du transistor
            # Capture: nom, w=VALUE+unit, middle, l=VALUE+unit
            pattern = (
                rf'^({re.escape(name)}\s+.*?w=)'  # Prefix avec "w="
                rf'([\d.eE+-]+)([umpn])'          # Ancienne valeur w + unité
                rf'(.*?l=)'                       # Middle avec "l="
                rf'([\d.eE+-]+)([umpn])'          # Valeur l + unité
            )
            
            def replacer(match):
                prefix = match.group(1)     # "X0 ... w="
                old_w_val = match.group(2)  # Ancienne valeur (ignorée)
                w_unit = match.group(3)     # Unité originale
                middle = match.group(4)     # " ... l="
                l_val = match.group(5)      # Longueur (inchangée)
                l_unit = match.group(6)     # Unité longueur
                
                # Convertir la nouvelle largeur dans l'unité d'origine
                new_w_val, _ = self._from_nanometers(params['w'], w_unit)
                
                # Formater selon la magnitude
                if new_w_val >= 1e5:
                    w_str = f"{new_w_val:.0f}"  # 700000
                elif new_w_val >= 1e3:
                    w_str = f"{new_w_val:.1f}"  # 12000.0
                else:
                    w_str = f"{new_w_val:.6g}"  # Notation adaptative
                
                return f"{prefix}{w_str}{w_unit}{middle}{l_val}{l_unit}"
            
            modified_content = re.sub(
                pattern,
                replacer,
                modified_content,
                flags=re.MULTILINE | re.IGNORECASE
            )
        
        # Sauvegarder
        if output_path is None:
            output_path = self.netlist_path
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(modified_content)
        
        return str(output_path)
    
    
    def reset(self) -> None:
        """
        Réinitialise toutes les modifications (recharge la netlist originale).
        """
        with open(self.netlist_path, 'r') as f:
            self.content = f.read()
        self.transistors = self._extract_transistors()
    
    
    def get_modification_summary(self) -> str:
        """
        Retourne un résumé lisible des transistors modifiables.
        
        Returns:
            Chaîne multiligne avec infos sur chaque transistor
        """
        lines = ["Transistors modifiables:"]
        for name, params in sorted(self.transistors.items()):
            lines.append(
                f"  {name}: W={params['w']:.1f}nm, L={params['l']:.1f}nm, "
                f"Type={params['type']}"
            )
        return '\n'.join(lines)
    
    
    def __repr__(self) -> str:
        """Représentation textuelle du modificateur."""
        return (
            f"CellModifier(netlist='{self.netlist_path.name}', "
            f"transistors={len(self.transistors)})"
        )


# ===== EXEMPLE D'UTILISATION =====
if __name__ == "__main__":
    """
    Test du modificateur avec une netlist d'inverseur.
    """
    print("🧪 Test de CellModifier\n")
    
    # 1. Charger une netlist
    try:
        modifier = CellModifier("/tmp/inv_test.sp")
        print("✅ Netlist chargée\n")
    except FileNotFoundError:
        print("❌ Netlist introuvable. Générez-la d'abord avec NetlistGenerator.")
        exit(1)
    
    # 2. Afficher l'état initial
    print(modifier.get_modification_summary())
    print(f"\n🔍 Largeurs initiales: {modifier.get_transistor_widths()}\n")
    
    # 3. Modifier les largeurs
    modifier.modify_width('X0', 700.0)   # NFET: 650 → 700nm
    modifier.modify_width('X1', 1200.0)  # PFET: 1000 → 1200nm
    
    print(f"🔍 Largeurs modifiées: {modifier.get_transistor_widths()}\n")
    
    # 4. Sauvegarder
    output = modifier.apply_modifications("/tmp/inv_modified.sp")
    print(f"✅ Netlist modifiée sauvegardée: {output}\n")
    
    # 5. Vérifier le contenu
    with open(output, 'r') as f:
        for line in f:
            if line.strip().startswith('X'):
                print(f"  {line.strip()}")
