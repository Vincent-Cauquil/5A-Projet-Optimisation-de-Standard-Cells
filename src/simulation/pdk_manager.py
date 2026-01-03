# src/simulation/pdk_manager.py 
"""
Auteurs : Vincent Cauquil (vincent.cauquil@cpe.fr)
          Léonard Anselme (leonard.anselme@cpe.fr)

Date : Novembre 2025 - Janvier 2026
"""
import subprocess
from pathlib import Path
from typing import Optional, List
import re

class PDKManager:
    """Gestionnaire pour localiser et utiliser les PDK"""

    DEFAULT_SEARCH_PATHS = [
        Path.home() / ".ciel",
        Path.home() / ".volare",
        Path("/usr/local/share/pdk"),
        Path("/opt/pdk"),
    ]

    def __init__(self, pdk_name: str = "sky130", use_uv: bool = True, verbose: bool = True):
        self.pdk_name = pdk_name
        self.use_uv = use_uv
        self.verbose = verbose 
        self._pdk_root = None
        self._lib_path = None
        self._cell_library = "sky130_fd_sc_hd"
        self._cdl_file = None
        self._cdl_content = None

    @property
    def pdk_root(self) -> Path:
        if self._pdk_root is None:
            self._pdk_root = self._find_pdk_root()
        return self._pdk_root

    @property
    def lib_path(self) -> Path:
        if self._lib_path is None:
            self._lib_path = self._find_lib_path()
        return self._lib_path

    @property
    def cdl_file(self) -> Path:
        """Fichier CDL contenant toutes les cellules"""
        if self._cdl_file is None:
            cdl = self.pdk_root / "libs.ref" / self._cell_library / "cdl" / f"{self._cell_library}.cdl"
            if not cdl.exists():
                raise FileNotFoundError(f"❌ CDL introuvable: {cdl}")
            self._cdl_file = cdl
            if self.verbose: 
                print(f"✓ CDL: {cdl}")
        return self._cdl_file

    def _find_pdk_root(self) -> Path:
        """Trouve le répertoire racine du PDK"""
        if self.verbose: 
            print(f"🔍 Recherche du PDK {self.pdk_name}...")

        for search_path in self.DEFAULT_SEARCH_PATHS:
            if not search_path.exists():
                continue

            pdk_dir = search_path / f"{self.pdk_name}A"
            if pdk_dir.exists() and (pdk_dir / "libs.tech").exists():
                if self.verbose:  
                    print(f"✓ PDK trouvé: {pdk_dir}")
                return pdk_dir

        raise FileNotFoundError(f"❌ PDK {self.pdk_name} introuvable")

    def _find_lib_path(self) -> Path:
        """Trouve le fichier principal de la librairie SPICE"""
        lib_file = self.pdk_root / "libs.tech" / "ngspice" / f"{self.pdk_name}.lib.spice"

        if not lib_file.exists():
            raise FileNotFoundError(f"❌ Librairie SPICE introuvable: {lib_file}")

        if self.verbose:  # ✅ Conditionnel
            print(f"✓ Librairie SPICE: {lib_file}")
        return lib_file

    def get_lib_include(self, corner: str = "tt") -> str:
        """
        Retourne la ligne .lib pour inclure le PDK

        Args:
            corner: Corner de process (tt, ff, ss, sf, fs)
        """
        lib_file = self.pdk_root / "libs.tech" / "ngspice" / "sky130.lib.spice"

        if not lib_file.exists():
            raise FileNotFoundError(f"Librairie SPICE introuvable: {lib_file}")

        if self.verbose:  # ✅ Conditionnel
            print(f"✓ Librairie SPICE: {lib_file}")

        # Format correct pour sky130
        return f".lib {lib_file.absolute()} {corner}"

    def clean_cell_spice(self, cell_content: str) -> str:
        """
        Nettoie le contenu SPICE d'une cellule pour compatibilité NGSpice

        Args:
            cell_content: Contenu brut de la cellule

        Returns:
            Contenu nettoyé
        """
        lines = []
        for line in cell_content.split('\n'):
            # Ignorer les lignes .ENDS internes et commentaires inutiles
            if line.strip().startswith('*') and 'Copyright' not in line:
                continue

            # Supprimer les paramètres problématiques
            line = line.replace('topography=normal', '')
            line = line.replace('area=0.063', '')
            line = line.replace('perim=1.14', '')

            # Nettoyer les espaces multiples
            line = re.sub(r'\s+', ' ', line)

            lines.append(line)

        return '\n'.join(lines)

    def extract_cell_from_cdl(self, cell_name: str, output_dir: Path) -> Path:
        """
        Extrait une cellule depuis le fichier CDL et la nettoie
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{cell_name}.spice"

        if output_file.exists():
            if self.verbose:  
                print(f"✓ Cellule déjà extraite: {output_file.name}")
            return output_file

        # Lire le CDL complet
        with open(self.cdl_file, 'r') as f:
            cdl_content = f.read()

        # Extraire la sous-circuit
        pattern = rf'\.SUBCKT\s+{re.escape(cell_name)}\s+.*?\.ENDS'
        match = re.search(pattern, cdl_content, re.DOTALL | re.IGNORECASE)

        if not match:
            raise ValueError(f"Cellule {cell_name} introuvable dans {self.cdl_file}")

        cell_spice = match.group(0)

        # Nettoyer le contenu
        cell_spice = self.clean_cell_spice(cell_spice)

        # Écrire dans un fichier
        with open(output_file, 'w') as f:
            f.write(f"* Extracted from CDL: {cell_name}\n")
            f.write(f"* Cleaned for NGSpice compatibility\n\n")
            f.write(cell_spice)
            f.write("\n")

        if self.verbose:  # ✅ Conditionnel
            print(f"✓ Cellule extraite et nettoyée: {output_file.name}")
        return output_file

    def get_complete_includes(self, corner: str = "tt") -> str:
        """
        Retourne toutes les includes nécessaires
        """
        lib_file = self.pdk_root / "libs.tech" / "ngspice" / "sky130.lib.spice"

        if not lib_file.exists():
            raise FileNotFoundError(f"Librairie SPICE introuvable: {lib_file}")

        return f".lib {lib_file.absolute()} {corner}"

    def _load_cdl(self):
        """Charge le contenu du CDL en mémoire"""
        if self._cdl_content is None:
            self._cdl_content = self.cdl_file.read_text()

    def list_available_cells(self, pattern: str = None) -> List[str]:
        """Liste les cellules disponibles dans le CDL"""
        self._load_cdl()

        cells = set()
        for match in re.finditer(r'\.SUBCKT\s+(\S+)', self._cdl_content, re.IGNORECASE):
            cell_name = match.group(1)

            if pattern is None or pattern.lower() in cell_name.lower():
                cells.add(cell_name)

        return sorted(cells)

    def get_cell_pins(self, cell_name: str) -> List[str]:
        """Extrait les noms de pins d'une cellule depuis le CDL"""
        if self._cdl_content is None:
            self._cdl_content = self.cdl_file.read_text()

        pattern = rf'\.SUBCKT\s+{re.escape(cell_name)}\s+([^\n]+)'
        match = re.search(pattern, self._cdl_content, re.IGNORECASE)

        if not match:
            raise ValueError(f"Cellule {cell_name} introuvable dans CDL")

        pins_line = match.group(1)
        pins = pins_line.split()

        return pins

    def get_cell_info(self, cell_name: str) -> dict:
        """Récupère les informations d'une cellule"""
        self._load_cdl()

        if not cell_name.startswith("sky130_"):
            cell_name = f"sky130_fd_sc_hd__{cell_name}_1"

        pattern = rf'\.SUBCKT\s+({re.escape(cell_name)})\s+(.*?)\n'
        match = re.search(pattern, self._cdl_content, re.IGNORECASE)

        if not match:
            raise ValueError(f"Cellule {cell_name} introuvable")

        ports = match.group(2).split()

        return {
            "name": cell_name,
            "ports": ports,
            "power_pins": [p for p in ports if p in ["VPWR", "VGND", "VPB", "VNB"]],
            "signal_pins": [p for p in ports if p not in ["VPWR", "VGND", "VPB", "VNB"]],
        }

