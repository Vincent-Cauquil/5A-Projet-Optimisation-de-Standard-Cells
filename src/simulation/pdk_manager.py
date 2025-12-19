# src/simulation/pdk_manager.py (mise à jour)
import subprocess
from pathlib import Path
from typing import Optional
import os

class PDKManager:
    """Gestionnaire pour localiser et utiliser les PDK"""
    
    # Chemins de recherche par défaut
    DEFAULT_SEARCH_PATHS = [
        Path("/usr/local/share/pdk"),
        Path.home() / ".volare",
        Path.home() / ".ciel",  # ← Ajout pour ciel
        Path("/opt/pdk"),
        Path(os.environ.get("PDK_ROOT", "/tmp")) if "PDK_ROOT" in os.environ else None,
    ]
    
    def __init__(self, pdk_name: str = "sky130", use_uv: bool = True):
        self.pdk_name = pdk_name
        self.use_uv = use_uv  # ← Flag pour utiliser uv
        self._pdk_root = None
        self._lib_path = None
        self._cell_library = "sky130_fd_sc_hd"
    
    @property
    def pdk_root(self) -> Path:
        """Retourne le répertoire racine du PDK"""
        if self._pdk_root is None:
            self._pdk_root = self._find_pdk_root()
        return self._pdk_root
    
    @property
    def lib_path(self) -> Path:
        """Retourne le chemin vers la librairie SPICE"""
        if self._lib_path is None:
            self._lib_path = self._find_lib_path()
        return self._lib_path
    
    def _find_pdk_root(self) -> Path:
        """Trouve le répertoire racine du PDK"""
        
        # Méthode 1: Via ciel (avec uv si nécessaire)
        try:
            # Construire la commande
            cmd = ['uv', 'run', 'ciel', 'get-pdk-dir', self.pdk_name] if self.use_uv else ['ciel', 'get-pdk-dir', self.pdk_name]
            
            print(f"🔍 Commande: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            pdk_path = Path(result.stdout.strip())
            
            if pdk_path.exists():
                print(f"✓ PDK trouvé via ciel: {pdk_path}")
                return pdk_path
            else:
                print(f"⚠ ciel a retourné un chemin invalide: {pdk_path}")
                
        except subprocess.CalledProcessError as e:
            print(f"⚠ Erreur ciel (code {e.returncode}): {e.stderr}")
        except subprocess.TimeoutExpired:
            print("⚠ Timeout ciel")
        except FileNotFoundError:
            print("⚠ ciel/uv non trouvé")
        
        # Méthode 2: Recherche manuelle
        print("🔍 Recherche manuelle du PDK...")
        
        for search_path in self.DEFAULT_SEARCH_PATHS:
            if search_path is None or not search_path.exists():
                continue
            
            print(f"  📁 Recherche dans {search_path}")
            
            # Chercher sky130A/B ou versions avec hash
            patterns = [
                f"{self.pdk_name}A",
                f"{self.pdk_name}B", 
                f"{self.pdk_name}a",
                self.pdk_name,
            ]
            
            for variant in patterns:
                # Recherche directe
                pdk_path = search_path / variant
                if pdk_path.exists() and self._is_valid_pdk(pdk_path):
                    print(f"✓ PDK trouvé: {pdk_path}")
                    return pdk_path
                
                # Recherche dans structure ciel (avec versions/hash)
                # Ex: ~/.ciel/ciel/sky130/versions/HASH/sky130A
                ciel_pattern = search_path / "ciel" / self.pdk_name / "versions"
                if ciel_pattern.exists():
                    # Chercher le dernier hash
                    version_dirs = sorted(ciel_pattern.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
                    for version_dir in version_dirs:
                        pdk_path = version_dir / variant
                        if pdk_path.exists() and self._is_valid_pdk(pdk_path):
                            print(f"✓ PDK trouvé (ciel): {pdk_path}")
                            return pdk_path
        
        # Méthode 3: Variable d'environnement
        if "PDK_ROOT" in os.environ:
            pdk_path = Path(os.environ["PDK_ROOT"]) / f"{self.pdk_name}A"
            if pdk_path.exists():
                print(f"✓ PDK trouvé via PDK_ROOT: {pdk_path}")
                return pdk_path
        
        raise FileNotFoundError(
            f"❌ Impossible de trouver le PDK {self.pdk_name}.\n"
            f"Chemins recherchés: {[str(p) for p in self.DEFAULT_SEARCH_PATHS if p]}\n"
            f"Vérifiez l'installation:\n"
            f"  uv run ciel list-pdks\n"
            f"  uv run ciel enable {self.pdk_name}\n"
            f"Ou définissez: export PDK_ROOT=/chemin/vers/pdk"
        )
    
    def _is_valid_pdk(self, path: Path) -> bool:
        """Vérifie qu'un répertoire est un PDK valide"""
        has_libs = (path / "libs.tech").exists() or (path / "libs.ref").exists()
        if has_libs:
            print(f"    ✓ PDK valide détecté")
        return has_libs
    
    def _find_lib_path(self) -> Path:
        """Trouve le fichier .lib.spice principal"""
        
        possible_paths = [
            self.pdk_root / "libs.tech" / "ngspice" / f"{self.pdk_name}.lib.spice",
            self.pdk_root / "libs.tech" / "ngspice" / f"{self.pdk_name}A.lib.spice",
            self.pdk_root / "libs.tech" / "combined" / f"{self.pdk_name}.lib.spice",
        ]
        
        for lib_file in possible_paths:
            if lib_file.exists():
                print(f"✓ Librairie SPICE: {lib_file}")
                return lib_file
        
        # Recherche récursive
        print("🔍 Recherche récursive de la librairie SPICE...")
        lib_files = list(self.pdk_root.rglob("*.lib.spice"))
        
        if lib_files:
            for lib_file in lib_files:
                if self.pdk_name in lib_file.name:
                    print(f"✓ Librairie SPICE: {lib_file}")
                    return lib_file
            
            print(f"✓ Librairie SPICE (première trouvée): {lib_files[0]}")
            return lib_files[0]
        
        raise FileNotFoundError(f"❌ Aucune librairie SPICE dans {self.pdk_root}")
    
    def get_lib_include(self, corner: str = "tt") -> str:
        """Retourne la ligne .lib à inclure dans une netlist"""
        return f".lib {self.lib_path} {corner}"
    
    def get_cell_spice(self, cell_name: str) -> Path:
        """Trouve le fichier SPICE d'une cellule"""
        # Ordre de recherche
        search_paths = [
            self.pdk_root / self._cell_library / "latest" / "cells" / cell_name,
            self.pdk_root / "libs.ref" / self._cell_library / "spice" / cell_name,
        ]
        
        for cell_dir in search_paths:
            if cell_dir.exists():
                spice_files = list(cell_dir.glob("*.spice"))
                if spice_files:
                    print(f"✓ Cellule {cell_name}: {spice_files[0]}")
                    return spice_files[0]
        
        # Recherche récursive
        matches = list(self.pdk_root.rglob(f"*{cell_name}*.spice"))
        if matches:
            print(f"✓ Cellule {cell_name}: {matches[0]}")
            return matches[0]
        
        raise FileNotFoundError(f"❌ Cellule {cell_name} introuvable dans {self.pdk_root}")
    
    def list_available_cells(self, pattern: str = "") -> list[str]:
        """Liste les cellules disponibles"""
        cells_dirs = [
            self.pdk_root / self._cell_library / "latest" / "cells",
            self.pdk_root / "libs.ref" / self._cell_library / "spice",
        ]
        
        cells = set()
        for cells_dir in cells_dirs:
            if cells_dir.exists():
                cells.update(d.name for d in cells_dir.iterdir() if d.is_dir())
        
        if pattern:
            cells = {c for c in cells if pattern.lower() in c.lower()}
        
        return sorted(cells)
