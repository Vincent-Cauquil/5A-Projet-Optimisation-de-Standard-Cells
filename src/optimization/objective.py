# src/optimization/objective.py
"""
Fonction objectif pour l'optimisation RL
Utilise CellModifier + NetlistGenerator + SpiceRunner
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import tempfile

from src.optimization.cell_modifier import CellModifier
from src.simulation.spice_runner import SpiceRunner
from src.simulation.pdk_manager import PDKManager
from src.simulation.netlist_generator import NetlistGenerator, SimulationConfig
from .simulation_cache import SimulationCache


class ObjectiveFunction:
    """Fonction objectif pour l'optimisation"""

    def __init__(
        self,
        cell_name: str,
        config: SimulationConfig,
        pdk: PDKManager,
        verbose: bool = False,
        use_cache: bool = True,
        penalty_cost: float = 1000.0
    ):
        self.cell_name = cell_name
        self.config = config
        self.pdk = pdk
        self.verbose = verbose
        self.penalty_cost = penalty_cost

        # Cache
        self.use_cache = use_cache
        self.cache = SimulationCache() if use_cache else None

        # ✅ Runner
        self.runner = SpiceRunner(
            pdk_root=pdk.pdk_root,
            verbose=verbose
        )

        # ✅ Generator (pour créer la netlist de base)
        self.generator = NetlistGenerator(pdk)

        # Cache des largeurs originales
        self._original_widths_cache: Optional[Dict[str, float]] = None
        self._base_netlist_path: Optional[Path] = None

        # Métriques de référence
        self.reference_metrics: Optional[Dict[str, float]] = None

        if self.verbose:
            print(f"✅ ObjectiveFunction initialisée pour {cell_name}")

    def get_original_widths(self) -> Dict[str, float]:
        """Retourne les largeurs originales de la cellule"""
        if self._original_widths_cache is not None:
            return self._original_widths_cache.copy()

        # ✅ Générer une netlist de caractérisation pour extraire les largeurs
        temp_dir = Path(tempfile.mkdtemp(prefix="obj_init_"))
        temp_netlist = temp_dir / f"{self.cell_name}_base.sp"

        try:
            # ✅ Utiliser generate_characterization_netlist
            self.generator.generate_characterization_netlist(
                cell_name=self.cell_name,
                output_path=str(temp_netlist),
                config=self.config
            )
            # Charger avec CellModifier
            modifier = CellModifier(str(temp_netlist))
            self._original_widths_cache = modifier.get_transistor_widths()
            self._base_netlist_path = temp_netlist

            if self.verbose:
                print(f"📏 Largeurs originales extraites:")
                for name, width in self._original_widths_cache.items():
                    print(f"   {name}: {width:.1f} nm")

            return self._original_widths_cache.copy()

        except Exception as e:
            if self.verbose:
                print(f"❌ Erreur extraction largeurs: {e}")
            raise


    def evaluate(
        self,
        widths: Dict[str, float],
        cost_weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Évalue les métriques pour un ensemble de largeurs
        """
        if cost_weights is None:
            cost_weights = {'delay': 0.5, 'energy': 0.3, 'area': 0.2}

        # Cache
        cache_key = None
        if self.use_cache and self.cache is not None:
            cache_key = self.cache.get_cache_key(widths, self.config)
            cached = self.cache.get(cache_key)
            if cached is not None:
                if self.verbose:
                    print("   ✅ Résultat en cache")
                return cached

        try:
            # ✅ 1. Générer netlist de base
            temp_dir = Path(tempfile.mkdtemp(prefix="rl_sim_"))
            base_netlist = temp_dir / f"{self.cell_name}_base.spice"

            self.generator.generate_characterization_netlist(
                cell_name=self.cell_name,
                output_path=str(base_netlist),
                config=self.config
            )

            # ✅ 2. Charger avec CellModifier
            modifier = CellModifier(str(base_netlist))

            # ✅ 3. Appliquer les modifications
            modifier.modify_multiple_widths(widths)

            # ✅ 4. Sauvegarder netlist modifiée
            modified_netlist = temp_dir / f"{self.cell_name}_modified.spice"
            modifier.apply_modifications(str(modified_netlist))

            if self.verbose:
                print(f"   📄 Netlist: {modified_netlist}")

            # ✅ 5. Simuler avec SpiceRunner
            result = self.runner.run_simulation(
                netlist_path=modified_netlist,  
                verbose=self.verbose
            )

            # ✅ 6. Vérifier succès
            sim_success = result.get('success', False)

            if not sim_success:
                if self.verbose:
                    errors = result.get('errors', ['Unknown error'])
                    print(f"   ❌ Simulation échouée: {errors[0]}")
                return self._penalty_result()

            # ✅ 7. Extraire métriques
            measures = result.get('measures', {})
            if not measures:
                if self.verbose:
                    print("   ❌ Aucune mesure extraite")
                return self._penalty_result()
            # ✅ 8. Calculer coût
            metrics = self._extract_metrics(measures, widths)
            cost = self._compute_cost(metrics, cost_weights)
            metrics['cost'] = cost

            # ✅ 9. Cache
            if self.use_cache and self.cache is not None and cache_key:
                self.cache.set(cache_key, metrics)
            

            return metrics

        except Exception as e:
            if self.verbose:
                print(f"   ❌ Erreur: {e}")
                import traceback
                traceback.print_exc()
            return self._penalty_result()

    def _extract_metrics(
        self,
        measures: Dict[str, float],
        widths: Dict[str, float]
    ) -> Dict[str, float]:
        """Extrait les métriques importantes des mesures NGSpice"""
        
        # Calculer delay_avg, delay_max, tplh_avg, tphl_avg
        tplh_values = [v for k, v in measures.items() if k.startswith('tplh_')]
        tphl_values = [v for k, v in measures.items() if k.startswith('tphl_')]
        
        if tplh_values and tphl_values:
            delay_avg = (np.mean(tplh_values) + np.mean(tphl_values)) / 2
            delay_max = max(max(tplh_values), max(tphl_values))
            tplh_avg = float(np.mean(tplh_values))
            tphl_avg = float(np.mean(tphl_values))
        else:
            delay_avg = float('inf')
            delay_max = float('inf')
            tplh_avg = float('inf')
            tphl_avg = float('inf')
        
        # Puissance et énergie
        power_avg = measures.get('power_avg', 0.0)
        energy_dyn = measures.get('energy_dyn', 0.0)
        
        # Aire
        area = sum(widths.values())
        
        return {
            'delay_avg': delay_avg,
            'delay_max': delay_max,
            'tplh_avg': tplh_avg,
            'tphl_avg': tphl_avg,
            'power_avg': power_avg,
            'energy_dyn': energy_dyn,
            'area': area
        }


    def _compute_cost(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calcule le coût pondéré à partir des métriques"""
        
        # Normaliser les métriques
        normalized = self._normalize_metrics(metrics)
        
        cost = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in normalized:
                cost += weight * normalized[metric]
                total_weight += weight
            elif self.verbose:
                print(f"⚠️  Métrique '{metric}' absente, ignorée")
        
        # Normaliser par la somme des poids
        if total_weight > 0:
            cost /= total_weight
        else:
            return self.penalty_cost
        
        return float(cost)
    
    def _normalize_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Normalise les métriques entre 0 et 1"""
        normalized = {}
        
        # Références typiques pour Sky130 inverter
        reference_values = {
            'delay_avg': 1e-10,      # 100ps
            'delay_max': 1.5e-10,    # 150ps
            'tplh_avg': 1e-10,       # 100ps
            'tphl_avg': 1e-10,       # 100ps
            'energy_dyn': 1e-14,     # 10fJ
            'power_avg': 1e-5,       # 10µW
            'area': 100.0            # 100 µm²
        }
        
        for key, value in metrics.items():
            if key in reference_values:
                ref = reference_values[key]
                if ref > 0 and value != float('inf'):
                    normalized[key] = min(value / ref, 10.0)  # Cap à 10x
                else:
                    normalized[key] = 10.0  # Pénalité maximale
            else:
                normalized[key] = value
        
        return normalized


    def _penalty_result(self) -> Dict[str, float]:
        """Retourne un résultat de pénalité"""
        return {
            'cost': self.penalty_cost,
            'delay_avg': float('inf'),
            'delay_max': float('inf'),
            'tplh_avg': float('inf'),
            'tphl_avg': float('inf'),
            'power_avg': 0.0,
            'energy_dyn': 0.0,
            'area': 0.0
        }
