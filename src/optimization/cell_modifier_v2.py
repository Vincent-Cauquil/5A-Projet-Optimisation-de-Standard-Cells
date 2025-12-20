# src/optimization/cell_modifier_v2.py
"""
Wrapper autour de CellModifier pour compatibilité RL
"""

from typing import Dict, Optional
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.cell_modifier import CellModifier  # ✅ Ancien CellModifier
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig


class RLCellModifier:
    """
    Wrapper autour de CellModifier pour l'optimisation RL
    Génère d'abord une netlist, puis utilise CellModifier dessus
    """

    def __init__(
        self,
        cell_name: str,
        pdk,
        verbose: bool = False
    ):
        self.cell_name = cell_name
        self.pdk = pdk
        self.verbose = verbose

        # Générateur de netlists
        self.netlist_gen = NetlistGenerator(pdk, verbose=verbose)

        # Cache de la netlist de base
        self._base_netlist_path: Optional[Path] = None
        self._modifier: Optional[CellModifier] = None
        self._original_widths: Optional[Dict[str, float]] = None

        if self.verbose:
            print(f"✅ RLCellModifier initialisé pour {cell_name}")

    def _ensure_base_netlist(self, config: SimulationConfig) -> Path:
        """
        Génère la netlist de base si pas déjà fait
        """
        if self._base_netlist_path is None or not self._base_netlist_path.exists():
            # Générer dans un répertoire temporaire
            temp_dir = Path(tempfile.gettempdir()) / "rl_netlists"
            temp_dir.mkdir(exist_ok=True)

            self._base_netlist_path = self.netlist_gen.generate_characterization_netlist(
                cell_name=self.cell_name,
                config=config,
                output_path=temp_dir / f"{self.cell_name}_base.sp"
            )

            # Créer le CellModifier sur cette netlist
            self._modifier = CellModifier(str(self._base_netlist_path))

            if self.verbose:
                print(f"   Netlist de base générée: {self._base_netlist_path}")

        return self._base_netlist_path

    def get_original_widths(self, config: SimulationConfig = None) -> Dict[str, float]:
        """
        Récupère les largeurs originales depuis la netlist générée
        
        Returns:
            Dict {transistor_name: width_nm}
        """
        if config is None:
            config = SimulationConfig()  # Config par défaut

        if self._original_widths is None:
            # Générer la netlist et extraire les largeurs
            self._ensure_base_netlist(config)
            self._original_widths = self._modifier.get_transistor_widths()

        # Retourner avec types Python natifs
        return {k: float(v) for k, v in self._original_widths.items()}

    def get_current_widths(self, config: SimulationConfig = None) -> Dict[str, float]:
        """Alias pour get_original_widths"""
        return self.get_original_widths(config)

    def generate_netlist(
        self,
        widths: Dict[str, float],
        config: SimulationConfig,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Génère une netlist avec les largeurs modifiées
        
        Args:
            widths: Dict {transistor_name: width_nm}
            config: Configuration de simulation
            output_path: Chemin de sortie
            
        Returns:
            Path de la netlist modifiée
        """
        # S'assurer qu'on a la netlist de base
        self._ensure_base_netlist(config)

        # Convertir en types Python natifs
        clean_widths = {k: float(v) for k, v in widths.items()}

        # Modifier les largeurs via CellModifier
        for trans_name, width in clean_widths.items():
            self._modifier.modify_width(trans_name, width)

        # Générer le fichier modifié
        if output_path is None:
            temp_dir = Path(tempfile.gettempdir()) / "rl_netlists"
            temp_dir.mkdir(exist_ok=True)
            output_path = temp_dir / f"{self.cell_name}_modified.sp"

        output_path = Path(output_path)
        modified_path = self._modifier.apply_modifications(str(output_path))

        # Réinitialiser le modifier pour la prochaine fois
        self._modifier.reset()

        if self.verbose:
            print(f"   Netlist modifiée: {modified_path}")

        return Path(modified_path)

    def validate_widths(
        self,
        widths: Dict[str, float],
        min_width: float = 420.0,
        max_width: float = 10000.0
    ) -> bool:
        """
        Valide les largeurs selon les DRC Sky130
        """
        for name, width in widths.items():
            width_float = float(width)

            if width_float < min_width or width_float > max_width:
                if self.verbose:
                    print(f"⚠️  {name}: {width_float:.0f}nm hors limites [{min_width}, {max_width}]")
                return False

        return True
