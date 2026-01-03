# ============================================================
#  CellModifier – Version SKY130 Standard‑Cell Compatible
# ============================================================
"""
Auteurs : Vincent Cauquil (vincent.cauquil@cpe.fr)
          Léonard Anselme (leonard.anselme@cpe.fr)

Date : Novembre 2025 - Janvier 2026
"""

import re
from pathlib import Path
from typing import Dict, Optional

class CellModifier:
    """
    Modifie les paramètres W/L des transistors MOSFET Sky130
    dans les netlists SPICE des standard cells.
    """

    MOS_PATTERN = re.compile(
        r"""
        ^
        (?P<name>[MX]\w+)                    # M0, X12, etc.
        \s+
        (?P<nodes>.*?)                       # noeuds
        \s+
        (?P<model>sky130_fd_pr__\S*fet\S*)   # nfet/pfet officiel
        (?P<params>.*)$
        """,
        re.IGNORECASE | re.MULTILINE | re.VERBOSE
    )

    WL_PATTERN = re.compile(
        r"(?P<key>[wl])\s*=\s*(?P<value>[0-9.eE+-]+)(?P<unit>[unp]?)",
        re.IGNORECASE
    )

    # ==========
    # UNITÉS : Sky130 standard cells
    # ==========

    def __init__(self, netlist_path: str, 
                 min_width_nm: float,
                 max_width_nm: float):
        self.netlist_path = Path(netlist_path)

        if not self.netlist_path.exists():
            raise FileNotFoundError(f"Netlist introuvable : {netlist_path}")

        self.min_width_pm = min_width_nm*1000
        self.max_width_pm = max_width_nm*1000
        self.content = self.netlist_path.read_text()
        self.transistors = self._extract_transistors()

        if not self.transistors:
            raise ValueError(
                f"Aucun transistor Sky130 trouvé dans : {netlist_path}"
            )

    # =========================================================
    # EXTRACTION
    # =========================================================
    def _extract_transistors(self) -> Dict[str, Dict]:
        trans = {}

        for match in self.MOS_PATTERN.finditer(self.content):
            name = match.group("name")
            model = match.group("model")
            params = match.group("params")

            w, l = self._parse_wl(params)

            if w is None or l is None:
                continue

            trans[name] = {
                "model": model,
                "w": w,      # nm
                "l": l,      # nm
                "raw_line": match.group(0)
            }

        return trans

    def _parse_wl(self, text: str):
        """Parse w=xxx l=yyy and convert to nm correctly."""

        w = None
        l = None

        for m in self.WL_PATTERN.finditer(text):
            key = m.group("key").lower()
            val = float(m.group("value"))

            if key == "w":
                w = val
            elif key == "l":
                l = val

        return w, l

    # =========================================================
    # MODIFICATION
    # =========================================================
    def modify_width(self, name: str, width: float):
        if name not in self.transistors:
            raise KeyError(f"Transistor {name} introuvable.")

        width = float(width)
        # Largeurs réalistes dans Sky130 HD
        if width < self.min_width_pm:
            self.transistors[name]["w"] = self.min_width_pm 
            raise ValueError(f"Largeur {width} pm < {self.min_width_pm}.")
        if width > self.max_width_pm:
            self.transistors[name]["w"] = self.max_width_pm 
            raise ValueError(f"Largeur {width} pm > {self.max_width_pm}.")

        self.transistors[name]["w"] = width 

    def modify_multiple_widths(self, widths: Dict[str, float]):
        for name, w in widths.items():
            self.modify_width(name, w/1e-12) #conversion en pm

    # =========================================================
    # ÉCRITURE
    # =========================================================
    def apply_modifications(self, output: Optional[str] = None) -> str:
        new_content = self.content

        for name, info in self.transistors.items():
            old = info["raw_line"]
            updated = self._update_line(old, info["w"], info["l"])
            new_content = new_content.replace(old, updated)

        out_path = Path(output) if output else self.netlist_path
        out_path.write_text(new_content)

        return str(out_path)

    def _update_line(self, line: str, w_meters: float, l_meters: float) -> str:
        """
        Remplace W et L dans la ligne.
        Entrées : w_meters, l_meters (floats en mètres, ex: 1.0e-6)
        Sortie  : w=1000000 l=150000 (entiers en picomètres, sans unité)
        """
        def _replace_param(match):
            key = match.group("key").lower()

            if key == "w": 
                val_m = w_meters
            else:
                val_m = l_meters
            
            val_str = f"{int(val_m)}" 
            
            return f"{key}={val_str}"


        return self.WL_PATTERN.sub(_replace_param, line)

    # =========================================================
    def get_transistor_widths(self):
        return {k: v["w"] for k, v in self.transistors.items()}

    def __repr__(self):
        return f"CellModifier({self.netlist_path.name}, {len(self.transistors)} transistors)"
