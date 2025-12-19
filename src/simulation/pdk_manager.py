# src/simulation/pdk_manager.py (FINAL avec CDL)
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
    
    def __init__(self, pdk_name: str = "sky130", use_uv: bool = True):
        self.pdk_name = pdk_name
        self.use_uv = use_uv
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
            print(f"✓ CDL: {cdl}")
        return self._cdl_file
    
    def _find_pdk_root(self) -> Path:
        """Trouve le répertoire racine du PDK"""
        print(f"🔍 Recherche du PDK {self.pdk_name}...")
        
        for search_path in self.DEFAULT_SEARCH_PATHS:
            if not search_path.exists():
                continue
            
            pdk_dir = search_path / f"{self.pdk_name}A"
            if pdk_dir.exists() and (pdk_dir / "libs.tech").exists():
                print(f"✓ PDK trouvé: {pdk_dir}")
                return pdk_dir
        
        raise FileNotFoundError(f"❌ PDK {self.pdk_name} introuvable")
    
    def _find_lib_path(self) -> Path:
        """Trouve le fichier principal de la librairie SPICE"""
        lib_file = self.pdk_root / "libs.tech" / "ngspice" / f"{self.pdk_name}.lib.spice"
        
        if not lib_file.exists():
            raise FileNotFoundError(f"❌ Librairie SPICE introuvable: {lib_file}")
        
        print(f"✓ Librairie SPICE: {lib_file}")
        return lib_file
    
    def get_lib_include(self, corner: str = "tt") -> str:
        """Génère l'instruction .lib pour SPICE"""
        return f".lib {self.lib_path} {corner}"
    
    def _load_cdl(self):
        """Charge le contenu du CDL en mémoire"""
        if self._cdl_content is None:
            self._cdl_content = self.cdl_file.read_text()
    
    def list_available_cells(self, pattern: str = None) -> List[str]:
        """Liste les cellules disponibles dans le CDL"""
        self._load_cdl()
        
        # Trouver toutes les .SUBCKT
        cells = set()
        for match in re.finditer(r'\.SUBCKT\s+(\S+)', self._cdl_content, re.IGNORECASE):
            cell_name = match.group(1)
            
            # Filtrer par pattern
            if pattern is None or pattern.lower() in cell_name.lower():
                cells.add(cell_name)
        
        return sorted(cells)
    
    def get_cell_spice(self, cell_name: str, output_dir: Optional[Path] = None) -> Path:
        """
        Extrait une cellule du CDL et la sauvegarde en SPICE
        
        Args:
            cell_name: Nom de base (ex: "xor2") ou complet (ex: "sky130_fd_sc_hd__xor2_1")
            output_dir: Répertoire de sortie (par défaut: ./netlists/cells/)
        """
        self._load_cdl()
        
        # Si nom court, chercher la variante _1
        if not cell_name.startswith("sky130_"):
            cell_name = f"sky130_fd_sc_hd__{cell_name}_1"
        
        # Pattern pour extraire la subckt complète
        pattern = rf'\.SUBCKT\s+{re.escape(cell_name)}\s+.*?\.ENDS\s+{re.escape(cell_name)}'
        
        match = re.search(pattern, self._cdl_content, re.DOTALL | re.IGNORECASE)
        
        if not match:
            raise FileNotFoundError(
                f"❌ Cellule {cell_name} introuvable dans {self.cdl_file}\n"
                f"💡 Cellules disponibles: {', '.join(self.list_available_cells(cell_name.split('__')[-1].split('_')[0])[:5])}"
            )
        
        # Créer le fichier SPICE
        if output_dir is None:
            output_dir = Path("netlists/cells")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{cell_name}.spice"
        
        spice_content = f"""* Extracted from {self.cdl_file.name}
* Cell: {cell_name}
* PDK: {self.pdk_name}

{match.group(0)}
"""
        
        output_file.write_text(spice_content)
        print(f"✓ Cellule extraite: {output_file.name}")
        
        return output_file
    
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
