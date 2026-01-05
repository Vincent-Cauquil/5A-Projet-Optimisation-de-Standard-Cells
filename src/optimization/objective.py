#!/usr/bin/env python3
# src/optimization/objective.py
# ============================================================
#  Objective Function
# ============================================================
"""
Fonction objectif pour l'optimisation par renforcement.
Gère la modification de netlist, l'appel à NGSpice et le calcul du coût.

Auteurs : Vincent Cauquil (vincent.cauquil@cpe.fr)
          Léonard Anselme (leonard.anselme@cpe.fr)

Date : Novembre 2025 - Janvier 2026

class ObjectiveFunction:
    evaluate : Évalue les métriques et le coût pour un set de largeurs.
    _extract_metrics : Analyse les mesures SPICE (tplh, tphl, power, etc.).
    _compute_cost : Calcule le coût normalisé par rapport à la baseline.
    _compute_area : Calcule l'aire active totale (Somme W*L).
"""

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import tempfile
import json

from src.models.weight_manager import WeightManager
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

        self.wm = WeightManager()
        
        # Cache
        self.use_cache = use_cache
        self.cache = SimulationCache() if use_cache else None
        
        # Runner et Generator
        self.runner = SpiceRunner(pdk_root=pdk.pdk_root, verbose=verbose)
        self.generator = NetlistGenerator(pdk)

        # On a besoin des longueurs (L) pour calculer l'aire dans _load_baseline
        specs = self.generator.extract_transistor_specs(cell_name)
        self.original_widths = {k: v['w'] for k, v in specs.items()}
        self.original_lengths = {k: v['l'] for k, v in specs.items()}
        
        # Maintenant on peut charger la baseline (qui appelle _compute_area)
        self.baseline = self._load_baseline_for_cell(cell_name)

        if self.verbose:
            print(f"✅ ObjectiveFunction initialisée pour {cell_name}")

    def evaluate(
        self,
        widths: Dict[str, float],
        cost_weights: Dict[str, float] = None,
        min_width_nm : float = 420,
        max_width_nm : float = 2_000_000,
    ) -> Dict[str, float]:
        """
        Évalue les métriques pour un ensemble de largeurs
        """
        if cost_weights is None:
            # Clés corrigées pour matcher le calcul de coût normalisé
            cost_weights = {
                'delay_avg_norm': 0.5,
                'energy_dyn_norm': 0.3, 
                'area_um2_norm': 0.2
            }

        # Cache
        cache_key = None
        if self.use_cache and self.cache is not None:
            cache_key = self.cache.get_cache_key(widths, self.config)
            cached = self.cache.get(cache_key)
            if cached is not None:
                if self.verbose:
                    print("   Résultat en cache")
                return cached

        try:
            # 1. Générer netlist de base
            temp_dir = Path(tempfile.mkdtemp(prefix="rl_sim_"))
            base_netlist = temp_dir / f"{self.cell_name}_base.spice"

            self.generator.generate_characterization_netlist(
                cell_name=self.cell_name,
                output_path=str(base_netlist),
                config=self.config
            )

            # 2. Charger avec CellModifier
            modifier = CellModifier(str(base_netlist), min_width_nm, max_width_nm)

            # 3. Appliquer les modifications
            modifier.modify_multiple_widths(widths)

            # 4. Sauvegarder netlist modifiée
            modified_netlist = temp_dir / f"{self.cell_name}_modified.spice"
            
            modifier.apply_modifications(str(modified_netlist))
            
            if self.verbose:
                print(f"   📄 Netlist: {modified_netlist}")

            # 5. Simuler avec SpiceRunner
            result = self.runner.run_simulation(
                netlist_path=modified_netlist,  
                verbose=self.verbose
            )

            # 6. Vérifier succès
            sim_success = result.get('success', False)

            if not sim_success:
                if self.verbose:
                    errors = result.get('errors', ['Unknown error'])
                    print(f"   ❌ Simulation échouée: {errors[0]}")
                return self._penalty_result()

            # 7. Extraire métriques
            measures = result.get('measures', {})
            if not measures:
                if self.verbose:
                    print("   ❌ Aucune mesure extraite")
                return self._penalty_result()
           
            
            metrics = self._extract_metrics(measures, widths)
            cost = self._compute_cost(metrics, cost_weights, self.cell_name) # Ajout cell_name manquant
            metrics['cost'] = cost
            
            if self.verbose :
                print(f"   ✔️ Simulation réussie - Coût: {cost:.4f}")
                print(f"     Métriques: {metrics}")
            if self.use_cache and self.cache is not None and cache_key:
                self.cache.set(cache_key, metrics)

            return metrics

        except Exception as e:
            if self.verbose:
                print(f"   ❌ Erreur: {e}")
                import traceback
                traceback.print_exc()
            return self._penalty_result()

    def _extract_metrics(self, measures: Dict[str, float], widths: Dict[str, float]) -> Dict[str, float]:
        """
        Extraction robuste pour multi-entrées, multi-transitions, multi-test.
        On regroupe les métriques par motifs (tplh, tphl, slew, power, etc.).
        """

        tplh = []
        tphl = []
        slew_in = []
        slew_out_rise = []
        slew_out_fall = []
        power_dyn = None
        leak_list = []

        for name, val in measures.items():
            low = name.lower()
            # --- DELAYS ---
            if "tplh" in low: tplh.append(val)
            if "tphl" in low: tphl.append(val)
            # --- SLEWS ---
            if "slew_in" in low: slew_in.append(val)
            if "slew_out" in low and "rise" in low: slew_out_rise.append(val)
            if "slew_out" in low and "fall" in low: slew_out_fall.append(val)
            # --- POWER DYN ---
            if "power_avg" in low: power_dyn = val
            # --- POWER LEAK (nouveau) ---
            if low.startswith("power_leak_t"): leak_list.append(val)

        # Fonctions d’agrégation
        def reduce_list(lst):
            return float(np.mean(lst)) if lst else float('inf')

        power_leak = float(np.mean(leak_list)) if leak_list else 0.0

        metrics = {
            'delay_rise': reduce_list(tplh),
            'delay_fall': reduce_list(tphl),
            'slew_in': reduce_list(slew_in),
            'slew_out_rise': reduce_list(slew_out_rise),
            'slew_out_fall': reduce_list(slew_out_fall),
            'power_dyn': power_dyn if power_dyn is not None else 0.0,
            'power_leak': power_leak,
            'area_um2': self._compute_area(widths)
        }
        metrics['delay_avg'] = (metrics['delay_rise'] + metrics['delay_fall']) / 2
        
        # energy_dyn est déjà en Joules dans power_dyn (c'est une intégrale de puissance)
        # Si SpiceRunner renvoie energy_dyn, on l'utilise directement
        if power_dyn != 0.0:
            metrics['energy_dyn'] = power_dyn
        else:
             metrics['energy_dyn'] = 0.0
 
        return metrics
    
    def _compute_cost(
        self,
        metrics: Dict[str, float],
        weights: Dict[str, float],
        cell_name: str
    ) -> float:
        """
        Calcule le coût RL pondéré.
        Exemple de weights : {'delay_avg_norm': 0.5, 'power_dyn_norm': 0.3, 'area_um2_norm': 0.2}
        """
        norm_metrics = self.get_normalized_metrics(cell_name, metrics)
        
        cost = 0.0
        total_weight = 0.0
        
        for m_norm, w in weights.items():
            if m_norm in norm_metrics:
                cost += w * norm_metrics[m_norm]
                total_weight += w
                
        # Le coût est le ratio pondéré moyen par rapport à la baseline
        return float(cost / total_weight) if total_weight > 0 else self.penalty_cost

    def _penalty_result(self) -> Dict[str, float]:
        """Retourne un résultat de pénalité"""
        return {
            'cost': self.penalty_cost,
            # Métriques agrégées
            'delay_avg': float('inf'),
            'power_avg': float('inf'), 
            'energy_dyn': float('inf'),
            'area_um2': float('inf'),
            
            # Métriques détaillées requises par StandardCellEnv
            'delay_rise': float('inf'),
            'delay_fall': float('inf'),
            'slew_in': float('inf'),
            'slew_out_rise': float('inf'),
            'slew_out_fall': float('inf'),
            'power_dyn': float('inf'),
            'power_leak': float('inf')
        }
    
    def _load_baseline_for_cell(self, cell_name: str) -> Dict:
        """
        Charge la baseline et la convertit au format 'metrics' standard.
        """
        category = self.wm._get_category(cell_name) 
        path = Path(f"src/models/references/{self.pdk.pdk_name}/{category}_baseline.json")
        
        if path.exists():
            with open(path, 'r') as f:
                all_data = json.load(f)
                cell_data = all_data.get(cell_name, {})
                
                if not cell_data:
                    return {}

                # Conversion des données brutes en format standard RL
                # Nécessite que self.original_lengths soit déjà initialisé
                raw_measures = cell_data.get('metrics', {})
                raw_widths = cell_data.get('widths', {}) 
                
                processed_metrics = self._extract_metrics(raw_measures, raw_widths)
                
                # On met à jour la baseline avec ces métriques standardisées
                cell_data['metrics'] = processed_metrics
                return cell_data
                
        return {}

    def get_normalized_metrics(self, cell_name: str, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Calcule les métriques normalisées : value / reference_baseline.
        Un ratio de 1.0 signifie que les performances sont identiques à l'original.
        """
        # Accès correct au sous-dictionnaire 'metrics' de la baseline
        ref_metrics = self.baseline.get('metrics', {})

        if not ref_metrics:
            if self.verbose: print(f"⚠️ Aucune baseline trouvée pour {cell_name}")
            return {f"{k}_norm": 1.0 for k in metrics.keys()}

        norm = {}
        for k, v in metrics.items():
            # On cherche la valeur de référence dans le dictionnaire ref_metrics
            if k in ref_metrics and ref_metrics[k] != 0:
                norm[f"{k}_norm"] = v / ref_metrics[k]
            else:
                norm[f"{k}_norm"] = 1.0
        return norm
    
    def _compute_area(self, widths: Dict[str, float]) -> float:
        """
        Calcule l'aire active totale des transistors (Somme W * L).
        Retourne la valeur en µm² (micromètres carrés).
        """
        total_area_m2 = 0.0
        
        for name, w_meters in widths.items():
            # Utilisation safe de original_lengths
            if self.original_lengths is not None:
                l_meters = self.original_lengths.get(name, 150e-9)
            else:
                # Fallback extreme (ne devrait pas arriver avec la correction __init__)
                l_meters = 150e-9 
            
            total_area_m2 += w_meters * l_meters

        # Conversion m² -> µm² (1 m² = 1e12 µm²)
        return total_area_m2 * 1e12