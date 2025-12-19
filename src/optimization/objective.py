# src/optimization/objective.py
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from ..simulation.netlist_generator import NetlistGenerator, SimulationConfig
from .cell_modifier import CellModifier
from .simulation_cache import SimulationCache
from ..simulation.spice_runner import SpiceRunner
from ..simulation.pdk_manager import PDKManager

class ObjectiveFunction:
    """
    Fonction objectif pour évaluer une configuration de transistors
    """

    def __init__(
        self,
        cell_name: str,
        config: SimulationConfig,
        pdk: PDKManager,
        use_cache: bool = True,  
        verbose: bool = False
    ):
        self.cell_name = cell_name
        self.config = config
        self.pdk = pdk
        self.verbose = verbose
        
        # ✅ Cache de simulation
        self.cache = SimulationCache(maxsize=1000) if use_cache else None

        # Générateur de netlist
        self.generator = NetlistGenerator(pdk)

        # Runner SPICE 
        self.runner = SpiceRunner(pdk.pdk_root)

        # Chemins temporaires
        self.netlist_path = Path(f"/tmp/{cell_name}_rl.sp")

        # Métriques de référence (cellule originale)
        self.reference_metrics = None
        self._compute_reference()

    def _compute_reference(self):
        """Calcule les métriques de la cellule originale"""
        if self.verbose:
            print("📊 Calcul des métriques de référence...")

        # Générer netlist originale
        netlist_str = self.generator.generate_characterization_netlist(
            cell_name=self.cell_name,
            config=self.config,
            output_path=str(self.netlist_path),
            
        )

        # Simuler avec run_simulation
        result = self.runner.run_simulation(self.netlist_path, verbose=False)

        if not result['success']:
            raise RuntimeError(f"Simulation de référence échouée: {result.get('errors', [])}")

        measures = result['measures']

        self.reference_metrics = {
            'delay_avg': measures.get('delay_avg', 1e-10),
            'energy_dyn': measures.get('energy_dyn', 1e-15),
            'power_avg': measures.get('power_avg', 1e-6),
        }

        if self.verbose:
            print(f"   Délai ref  : {self.reference_metrics['delay_avg']*1e12:.2f} ps")
            print(f"   Énergie ref: {self.reference_metrics['energy_dyn']*1e15:.2f} fJ")

    def evaluate(
        self,
        widths: Dict[str, float],
        weights: Dict[str, float] = None
    ) -> Tuple[float, Dict]:
        """
        Évalue une configuration de largeurs avec cache

        Args:
            widths: {'X0': 700.0, 'X1': 1200.0} en nm
            weights: {'delay': 0.5, 'energy': 0.3, 'area': 0.2}

        Returns:
            (cost, metrics_dict)
        """
        if weights is None:
            weights = {'delay': 0.5, 'energy': 0.3, 'area': 0.2}

        # ✅ Vérifier le cache d'abord
        if self.cache:
            found, cached_measures = self.cache.get(widths)
            if found:
                # Recalculer le coût avec les mesures cachées
                cost, metrics = self._compute_cost(
                    cached_measures, 
                    widths, 
                    weights
                )
                return cost, metrics

        # Pas dans le cache: simuler
        measures = self._simulate_widths(widths)
        
        if not measures:
            return float('inf'), {}

        # ✅ Stocker dans le cache
        if self.cache:
            self.cache.put(widths, measures)

        # Calculer le coût
        cost, metrics = self._compute_cost(measures, widths, weights)
        
        return cost, metrics

    def _simulate_widths(self, widths: Dict[str, float]) -> Dict[str, float]:
        """
        Simule une configuration de largeurs
        
        Returns:
            measures dict ou {} si échec
        """
        # 1. Générer netlist de base
        self.generator.generate_characterization_netlist(
            cell_name=self.cell_name,
            config=self.config,
            output_path=str(self.netlist_path),
            
        )

        # 2. Modifier les largeurs
        modifier = CellModifier(self.netlist_path)
        for trans_name, width_nm in widths.items():
            modifier.modify_width(trans_name, width_nm)

        output_path = self.netlist_path.with_suffix('.modified.sp')
        modifier.apply_modifications(str(output_path))

        # 3. Simuler
        try:
            result = self.runner.run_simulation(output_path, verbose=False)

            if not result['success']:
                if self.verbose:
                    print(f"❌ Simulation échouée: {result.get('errors', [])}")
                return {}

            return result['measures']

        except Exception as e:
            if self.verbose:
                print(f"❌ Exception durant simulation: {e}")
            return {}

    def _compute_cost(
        self,
        measures: Dict[str, float],
        widths: Dict[str, float],
        weights: Dict[str, float]
    ) -> Tuple[float, Dict]:
        """
        Calcule le coût à partir des mesures
        
        Args:
            measures: Résultats de simulation
            widths: Configuration de largeurs
            weights: Poids des objectifs
            
        Returns:
            (cost, metrics_dict)
        """
        # Extraire métriques
        delay = measures.get('delay_avg', float('inf'))
        energy = measures.get('energy_dyn', float('inf'))
        power = measures.get('power_avg', float('inf'))

        # Calculer l'aire (approximation)
        area = sum(widths.values())  # nm (proportionnel)

        # Normaliser par rapport à la référence
        delay_norm = delay / self.reference_metrics['delay_avg']
        energy_norm = energy / self.reference_metrics['energy_dyn']
        area_norm = area / sum(self._get_original_widths().values())

        # Calculer le coût pondéré
        cost = (
            weights['delay'] * delay_norm +
            weights['energy'] * energy_norm +
            weights['area'] * area_norm
        )

        metrics = {
            'delay_avg': delay,
            'energy_dyn': energy,
            'power_avg': power,
            'area': area,
            'delay_norm': delay_norm,
            'energy_norm': energy_norm,
            'area_norm': area_norm,
            'cost': cost
        }

        return cost, metrics

    def _get_original_widths(self) -> Dict[str, float]:
        """Retourne les largeurs originales"""
        # Cache pour éviter de re-parser
        if not hasattr(self, '_original_widths_cache'):
            # Régénérer la netlist originale si nécessaire
            if not self.netlist_path.exists():
                self.generator.generate_characterization_netlist(
                    cell_name=self.cell_name,
                    config=self.config,
                    output_path=str(self.netlist_path),
                    
                )

            modifier = CellModifier(self.netlist_path)
            self._original_widths_cache = modifier.get_transistor_widths()
        
        return self._original_widths_cache

    def get_cache_stats(self) -> Optional[Dict]:
        """Retourne les stats du cache si activé"""
        if self.cache:
            return self.cache.stats()
        return None
