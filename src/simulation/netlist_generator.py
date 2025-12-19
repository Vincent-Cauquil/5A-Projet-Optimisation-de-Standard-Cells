# src/simulation/netlist_generator.py
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass
import re

@dataclass
class SimulationConfig:
    """Configuration de simulation"""
    vdd: float = 1.8
    temp: float = 27
    corner: str = "tt"
    cload: float = 10e-15  # 10fF
    trise: float = 100e-12
    tfall: float = 100e-12

@dataclass
class GateLogic:
    """Définit la logique d'une porte"""
    function: Callable
    transition_states: Dict[str, str]

@dataclass
class TransitionTest:
    """Définit un test de transition"""
    name: str
    input_signals: Dict[str, str]
    measures: List[str]

class NetlistGenerator:
    """Génère des netlists SPICE pour caractérisation"""

    def __init__(self, pdk_manager, output_dir: Optional[Path] = None):
        self.pdk = pdk_manager

        if output_dir is None:
            self.output_dir = self.pdk.pdk_root / "libs.tech" / "ngspice"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lib_spice = self.pdk.pdk_root / "libs.ref" / "sky130_fd_sc_hd" / "spice" / "sky130_fd_sc_hd.spice"
        
        # Définition des portes logiques
        self.gate_logic = {
            'inv': GateLogic( # Porte INVERTEUR
                function=lambda a: not a,
                transition_states={'enable': {}}
            ),
            'buf': GateLogic( # Porte BUFFER
                function=lambda a: a,
                transition_states={'enable': {}}
            ),
            'nand': GateLogic( # Porte NAND avec plusieurs entrées
                function=lambda *inputs: not all(inputs),
                transition_states={'enable': {'others': '1'}}
            ),
            'nor': GateLogic( # Porte NOR avec plusieurs entrées
                function=lambda *inputs: not any(inputs),
                transition_states={'enable': {'others': '0'}}
            ),
            'and': GateLogic( # Porte AND avec plusieurs entrées
                function=lambda *inputs: all(inputs),
                transition_states={'enable': {'others': '1'}}
            ),
            'or': GateLogic( # Porte OR avec plusieurs entrées
                function=lambda *inputs: any(inputs),
                transition_states={'enable': {'others': '0'}}
            ),
            'xor': GateLogic( # XOR avec plusieurs entrées
                function=lambda *inputs: sum(inputs) % 2 == 1,
                transition_states={'enable': {'others': '0'}}
            ),
            'xnor': GateLogic( # XNOR = NOT(XOR)
                function=lambda *inputs: sum(inputs) % 2 == 0,  
                transition_states={'enable': {'others': '0'}}   
            ),
        }

    def _identify_inverted_inputs(self, cell_name: str, inputs: List[str]) -> Dict[str, bool]:
        """
        Identifie quelles entrées sont inversées dans le nom de la cellule par :
        1. Le suffixe '_N' dans le nom du port (prioritaire)
        2. La convention Sky130 (dernières lettres avec 'b')
        Exemples:
        - or2b_1 → B inversé → {'A': False, 'B': True}
        - or3b_1 → B inversé → {'A': False, 'B': True, 'C': False}
        - and2bb_1 → A et B inversés → {'A': True, 'B': True}        
        """
        inverted = {inp: False for inp in inputs}
        
        for inp in inputs:
            if inp.upper().endswith('_N'):
                inverted[inp] = True
        
        # Si des inversions trouvées par _N, on s'arrête là
        if any(inverted.values()):
            return inverted
        
        # Convention Sky130
        #  Par convention SKY130: les dernières entrées sont inversées
        #   or2b_1 : A OR ~B (B est inversé)
        #   or3b_1 : A OR B OR ~C (C est inversé)
        #   and2bb_1 : ~A AND ~B (A et B inversés) 
        cell_lower = cell_name.lower()
        for gate in ['nand', 'nor', 'and', 'or', 'xor', 'xnor']:
            if gate in cell_lower:
                pattern = rf'{gate}(\d+)(b+)'
                match = re.search(pattern, cell_lower)
                
                if match:
                    n_inverted = len(match.group(2))
                    
                    if n_inverted >= len(inputs):
                        for inp in inputs:
                            inverted[inp] = True
                    elif n_inverted > 0:
                        for i in range(n_inverted):
                            inv_idx = len(inputs) - n_inverted + i
                            if inv_idx < len(inputs):
                                inverted[inputs[inv_idx]] = True
                break
        
        return inverted

    def _get_base_gate_type(self, cell_name: str) -> str:
        """Extrait le type de porte de base (sans considérer les inversions)"""
        cell_lower = cell_name.lower()
        
        # Ordre important: composés d'abord
        for gate in ['xnor', 'xor', 'nand', 'nor', 'and', 'or', 'buf', 'inv']:
            if gate in cell_lower:
                return gate
        
        raise ValueError(f"Type de porte non reconnu: {cell_name}")

    def _get_gate_type(self, cell_name: str) -> str:
        """Extrait le type de porte (alias pour compatibilité)"""
        return self._get_base_gate_type(cell_name)

    def _parse_transition(self, signal: str) -> Tuple[str, str]:
        """Parse '0→1', '1→0' ou '1'"""
        if "→" in signal:
            parts = signal.split("→")
            if parts[0] == "0" and parts[1] == "1":
                return ("rise", "0")
            elif parts[0] == "1" and parts[1] == "0":
                return ("fall", "1")
        return ("static", signal)

    def generate_netlist(
        self,
        cell_name: str,
        config: SimulationConfig = SimulationConfig(),
        transitions: Optional[List[TransitionTest]] = None
    ) -> Path:
        """Génère une netlist de caractérisation """

        output_file = self.output_dir / f"{cell_name}_delay.spice"

        if transitions is None:
            transitions = self._generate_default_transitions(cell_name)

        ports_info = self._get_cell_ports(cell_name)
        input_pins = ports_info['input_list']
        output_pin = ports_info['output']
        all_ports_ordered = ports_info['all_ports']

        # ===== EN-TÊTE =====
        netlist_lines = [
            f"* Cell Characterization: {cell_name}",
            f"* Generated by NetlistGenerator",
            f"* Corner: {config.corner}, VDD={config.vdd}V, Temp={config.temp}°C",
            "",
            "* ===== CONVERGENCE OPTIONS =====",
            ".option reltol=1e-3",
            ".option abstol=1e-12",
            ".option vntol=1e-6",
            ".option gmin=1e-15",
            ".option method=gear",
            "",
            "* ===== PDK LIBRARY =====",
            f".lib sky130.lib.spice {config.corner}",
            "",
            "* ===== CELL LIBRARY =====",
            f".include {self.lib_spice}",
            "",
            "* ===== PARAMETERS =====",
            f".param SUPPLY={config.vdd}",
            f".param CLOAD={config.cload}",
            f".param TRISE={config.trise}",
            f".param TFALL={config.tfall}",
            "",
            f".temp {config.temp}",
            "",
        ]

        # ===== ALIMENTATIONS =====
        netlist_lines.extend([
            "* ===== POWER SUPPLIES =====",
            "Vdd vdd 0 DC {SUPPLY}",
            "Vss vss 0 DC 0",
            "Vpb vpb 0 DC {SUPPLY}",
            "Vnb vnb 0 DC 0",
            "",
        ])

        # ===== GÉNÉRATION PWL =====
        test_duration = 2e-9
        settling_time = 1e-9
        
        pin_states = {pin: 0.0 for pin in input_pins}
        pwl_data = {pin: [(0, 0.0)] for pin in input_pins}

        for test_idx, test in enumerate(transitions):
            test_start = test_idx * (test_duration + settling_time)
            transition_time = test_start + 0.5e-9

            for pin in input_pins:
                signal = test.input_signals.get(pin, None)
                if signal is None:
                    continue

                trans_type, target = self._parse_transition(signal)

                if trans_type == "static":
                    target_voltage = float(target) * config.vdd
                    if pin_states[pin] != target_voltage:
                        self._safe_add(pwl_data[pin], test_start, pin_states[pin])
                        self._safe_add(pwl_data[pin], test_start + 1e-12, target_voltage)
                        pin_states[pin] = target_voltage

                elif trans_type == "rise":
                    if pin_states[pin] != 0:
                        self._safe_add(pwl_data[pin], test_start, pin_states[pin])
                        self._safe_add(pwl_data[pin], test_start + 1e-12, 0.0)

                    self._safe_add(pwl_data[pin], transition_time, 0.0)
                    self._safe_add(pwl_data[pin], transition_time + config.trise, config.vdd)
                    pin_states[pin] = config.vdd

                elif trans_type == "fall":
                    if pin_states[pin] != config.vdd:
                        self._safe_add(pwl_data[pin], test_start, pin_states[pin])
                        self._safe_add(pwl_data[pin], test_start + 1e-12, config.vdd)

                    self._safe_add(pwl_data[pin], transition_time, config.vdd)
                    self._safe_add(pwl_data[pin], transition_time + config.tfall, 0.0)
                    pin_states[pin] = 0.0

            test_end = test_start + test_duration
            for pin in input_pins:
                self._safe_add(pwl_data[pin], test_end, pin_states[pin])

        # ===== SOURCES PWL =====
        netlist_lines.append("* ===== INPUT SIGNALS (PWL) =====")

        for pin in input_pins:
            pwl_parts = []
            prev_v = None
            
            for t, v in pwl_data[pin]:
                t_ns = round(t * 1e9, 3)  # ← Arrondir à 3 décimales
                
                if prev_v is not None and v != prev_v:
                    # Début de transition
                    pwl_parts.append(f"{t_ns:.3f}n {prev_v:.1f}")
                    
                    # Fin de transition avec slew
                    slew_param = "TRISE" if v > prev_v else "TFALL"
                    pwl_parts.append(f"'{t_ns:.3f}n+{slew_param}' {v:.1f}")
                else:
                    # Niveau stable
                    pwl_parts.append(f"{t_ns:.3f}n {v:.1f}")
                
                prev_v = v
            
            pwl_str = " ".join(pwl_parts)
            netlist_lines.append(f"V{pin} {pin.lower()} 0 PWL({pwl_str})")

        netlist_lines.append("")

        # ===== INSTANCIATION DUT =====
        dut_connections = []
        for port in all_ports_ordered:
            p_lower = port.lower()
            if p_lower in ['vpwr', 'vdd']:
                dut_connections.append('vdd')
            elif p_lower in ['vgnd', 'vss']:
                dut_connections.append('vss')
            elif p_lower == 'vpb':
                dut_connections.append('vpb')
            elif p_lower == 'vnb':
                dut_connections.append('vnb')
            else:
                dut_connections.append(p_lower)

        netlist_lines.extend([
            "* ===== DEVICE UNDER TEST =====",
            f"XCELL {' '.join(dut_connections)} {cell_name}",
            "",
            "* ===== OUTPUT LOAD =====",
            f"CL {output_pin.lower()} 0 {{CLOAD}}",
            "",
        ])

        # ===== MESURES DE DÉLAI =====
        netlist_lines.append("* ===== DELAY MEASUREMENTS =====")
        threshold = config.vdd / 2

        for test_idx, test in enumerate(transitions):
            t_start = test_idx * (test_duration + settling_time)
            t_end = t_start + test_duration

            netlist_lines.append(f"* Test {test_idx + 1}: {test.name}")

            for measure in test.measures:
                # Remplacer {{SUPPLY/2}}
                adapted = measure.replace("{{SUPPLY/2}}", str(threshold))
                
                # ✅ Rendre le nom unique en ajoutant _t{numero}
                parts = adapted.split()
                if len(parts) >= 3 and parts[0] == '.meas':
                    # parts[0] = '.meas'
                    # parts[1] = 'tran'
                    # parts[2] = 'tplh' ou 'tphl'
                    measure_name = parts[2]
                    unique_name = f"{measure_name}_t{test_idx+1}"
                    parts[2] = unique_name
                    adapted = ' '.join(parts)
                
                # Ajouter FROM/TO
                adapted += f" FROM={t_start*1e9:.3f}n TO={t_end*1e9:.3f}n"
                netlist_lines.append(adapted)

            netlist_lines.append("")

        # ===== MESURES DE CONSOMMATION ===== 
        netlist_lines.append("* ===== POWER MEASUREMENTS =====")

        total_time = len(transitions) * (test_duration + settling_time)
        total_time_ns = total_time * 1e9

        # Énergie dynamique totale
        netlist_lines.append(
            f".meas TRAN energy_dyn INTEG PAR('v(vdd) * -i(Vdd)') "
            f"FROM=0n TO={total_time_ns:.3f}n"
        )

        # Énergie par transition (optionnel mais utile)
        for test_idx, test in enumerate(transitions):
            t_start = test_idx * (test_duration + settling_time)
            t_end = t_start + test_duration

            netlist_lines.append(
                f".meas TRAN energy_test{test_idx+1} INTEG PAR('v(vdd) * -i(Vdd)') "
                f"FROM={t_start*1e9:.3f}n TO={t_end*1e9:.3f}n"
            )


        # ===== SIMULATION =====
        netlist_lines.extend([
            "* ===== TRANSIENT ANALYSIS =====",
            f".tran 1p {total_time_ns:.3f}n",
        ])
        # ===== SECTION CONTROL =====
        netlist_lines.extend([
            "",
            ".control",
            "run",
            "",
            "* Affichage des mesures",
            "echo \"\"",
            "echo \"===== MEASUREMENT RESULTS =====\"",
            "echo \"\"",
            "",
            "echo \"\"",
            "echo \"===== ENERGY RESULTS =====\"",
            "echo \"\"",
            "print energy_dyn power_avg energy_test1 energy_test2 energy_test3 energy_test4",
            "",
            ".endc",
            "",
            ".end"
        ])

        output_file.write_text("\n".join(netlist_lines))
        return output_file

    def _safe_add(self, pts: List[Tuple[float, float]], t: float, v: float):
        """Ajoute un point PWL uniquement si t est croissant"""
        if not pts or pts[-1][0] < t:
            pts.append((t, v))

    def _get_cell_ports(self, cell_name: str) -> dict:
        """Extrait les ports d'une cellule du fichier SPICE"""
        with open(self.lib_spice, 'r') as f:
            lines = f.readlines()

        in_cell = False
        ports_lines = []

        for line in lines:
            line = line.strip()
            if line.lower().startswith(f'.subckt {cell_name.lower()}'):
                in_cell = True
                parts = line.split(maxsplit=2)
                if len(parts) > 2:
                    ports_lines.append(parts[2])
                continue
            if in_cell and line.startswith('+'):
                ports_lines.append(line[1:].strip())
                continue
            if in_cell:
                break

        if not ports_lines:
            raise ValueError(f"Cellule {cell_name} introuvable dans {self.lib_spice}")

        ports_text = ' '.join(ports_lines)
        all_ports = ports_text.split()

        power_ports = {'VPWR', 'VGND', 'VPB', 'VNB', 'VDD', 'VSS'}
        signal_ports = [p for p in all_ports if p.upper() not in power_ports]

        output_port = signal_ports[-1]
        input_ports = signal_ports[:-1]

        return {
            'inputs': ' '.join(input_ports),
            'output': output_port,
            'all_ports': all_ports,
            'input_list': input_ports
        }

    def _generate_default_transitions(self, cell_name: str) -> List[TransitionTest]:
        """Génère les tests par défaut de manière générique"""
        try:
            ports = self._get_cell_ports(cell_name)
            inputs = ports['input_list']
            output = ports['output']
            gate_type = self._get_base_gate_type(cell_name)
            logic = self.gate_logic[gate_type]
            
            # Identifier les entrées inversées
            inverted_map = self._identify_inverted_inputs(cell_name, inputs)
            
        except Exception as e:
            print(f"⚠️  Erreur génération transitions pour {cell_name}: {e}")
            return []

        n_inputs = len(inputs)

        if n_inputs == 1:
            is_inverted = inverted_map.get(inputs[0], False)
            return self._generate_single_input_transitions(
                inputs[0], output, logic, is_inverted
            )
        else:
            return self._generate_multi_input_transitions(
                inputs, output, logic, inverted_map
            )

    def _generate_single_input_transitions(
        self, 
        input_pin: str, 
        output_pin: str, 
        logic: GateLogic,
        is_inverted: bool = False
    ) -> List[TransitionTest]:
        """Génère transitions pour porte 1 entrée"""
        
        # Si l'entrée est inversée, inverser les valeurs logiques
        if is_inverted:
            out_0 = logic.function(True)   # Input physique 0 = logique 1
            out_1 = logic.function(False)  # Input physique 1 = logique 0
        else:
            out_0 = logic.function(False)
            out_1 = logic.function(True)

        # Rise transition (0→1 physique)
        metric_rise = "tplh" if out_1 else "tphl"
        targ_edge_rise = "RISE" if out_1 else "FALL"

        # Fall transition (1→0 physique)
        metric_fall = "tphl" if not out_0 else "tplh"
        targ_edge_fall = "FALL" if not out_0 else "RISE"

        return [
            TransitionTest(
                name=f"{input_pin}: 0→1 → {output_pin}: {int(out_0)}→{int(out_1)} ({metric_rise})",
                input_signals={input_pin: "0→1"},
                measures=[
                    f".meas tran {metric_rise} TRIG v({input_pin.lower()}) VAL='{{{{SUPPLY/2}}}}' RISE=1 "
                    f"TARG v({output_pin.lower()}) VAL='{{{{SUPPLY/2}}}}' {targ_edge_rise}=1"
                ]
            ),
            TransitionTest(
                name=f"{input_pin}: 1→0 → {output_pin}: {int(out_1)}→{int(out_0)} ({metric_fall})",
                input_signals={input_pin: "1→0"},
                measures=[
                    f".meas tran {metric_fall} TRIG v({input_pin.lower()}) VAL='{{{{SUPPLY/2}}}}' FALL=1 "
                    f"TARG v({output_pin.lower()}) VAL='{{{{SUPPLY/2}}}}' {targ_edge_fall}=1"
                ]
            )
        ]

    def _generate_multi_input_transitions(
        self,
        inputs: List[str],
        output: str,
        logic: GateLogic,
        inverted_map: Dict[str, bool]
    ) -> List[TransitionTest]:
        """Génère transitions pour porte N entrées avec support des inversions"""

        transitions = []
        n_inputs = len(inputs)
        enable_state_logical = logic.transition_states['enable'].get('others', '0')

        for idx, active_input in enumerate(inputs):
            is_active_inverted = inverted_map.get(active_input, False)
            
            # ✅ CORRECTION : Calculer l'état physique selon l'inversion
            other_inputs = {}
            for inp in inputs:
                if inp != active_input:
                    is_other_inverted = inverted_map.get(inp, False)
                    
                    # État logique désiré (de la définition de la porte)
                    logical_state = enable_state_logical
                    
                    # État physique à appliquer (inverser si l'entrée est inversée)
                    if is_other_inverted:
                        # Si on veut logical=1 et que l'entrée est inversée, mettre phys=0
                        physical_state = '0' if logical_state == '1' else '1'
                    else:
                        physical_state = logical_state
                    
                    other_inputs[inp] = physical_state

            # === CALCUL DES ÉTATS LOGIQUES ===
            def build_logical_state(physical_states: List[bool]) -> List[bool]:
                """Convertit états physiques en états logiques"""
                logical = []
                for i, inp in enumerate(inputs):
                    phys_val = physical_states[i]
                    if inverted_map.get(inp, False):
                        logical.append(not phys_val)
                    else:
                        logical.append(phys_val)
                return logical

            # === ÉTAT INITIAL : active_input=0 (physique), others=enable (logique) ===
            phys_initial = []
            for i, inp in enumerate(inputs):
                if i == idx:
                    # L'entrée active commence à 0 (physique)
                    phys_initial.append(False)
                else:
                    # Les autres entrées à l'état d'activation (logique)
                    is_other_inv = inverted_map.get(inp, False)
                    logical_val = (enable_state_logical == '1')
                    
                    if is_other_inv:
                        # Inverser pour obtenir la valeur logique désirée
                        phys_initial.append(not logical_val)
                    else:
                        phys_initial.append(logical_val)
            
            logic_initial = build_logical_state(phys_initial)
            out_initial = logic.function(*logic_initial)

            # === ÉTAT APRÈS RISE : active_input=1 ===
            phys_after_rise = phys_initial.copy()
            phys_after_rise[idx] = True
            logic_after_rise = build_logical_state(phys_after_rise)
            out_after_rise = logic.function(*logic_after_rise)

            # === ÉTAT AVANT FALL : active_input=1 ===
            phys_before_fall = []
            for i, inp in enumerate(inputs):
                if i == idx:
                    phys_before_fall.append(True)
                else:
                    is_other_inv = inverted_map.get(inp, False)
                    logical_val = (enable_state_logical == '1')
                    
                    if is_other_inv:
                        phys_before_fall.append(not logical_val)
                    else:
                        phys_before_fall.append(logical_val)
            
            logic_before_fall = build_logical_state(phys_before_fall)
            out_before_fall = logic.function(*logic_before_fall)

            # === ÉTAT APRÈS FALL : active_input=0 ===
            phys_after_fall = phys_initial.copy()
            logic_after_fall = build_logical_state(phys_after_fall)
            out_after_fall = logic.function(*logic_after_fall)

            # === TRANSITION RISE (0→1 physique) ===
            if out_initial != out_after_rise:
                metric = "tplh" if out_after_rise else "tphl"
                targ_edge = "RISE" if out_after_rise else "FALL"

                inv_marker = "~" if is_active_inverted else ""
                transitions.append(TransitionTest(
                    name=f"{active_input}: 0→1{inv_marker}, others={enable_state_logical}(log) → {output}: {int(out_initial)}→{int(out_after_rise)} ({metric})",
                    input_signals={active_input: "0→1", **other_inputs},
                    measures=[
                        f".meas tran {metric} TRIG v({active_input.lower()}) VAL='{{{{SUPPLY/2}}}}' RISE=1 "
                        f"TARG v({output.lower()}) VAL='{{{{SUPPLY/2}}}}' {targ_edge}=1"
                    ]
                ))

            # === TRANSITION FALL (1→0 physique) ===
            if out_before_fall != out_after_fall:
                metric = "tphl" if not out_after_fall else "tplh"
                targ_edge = "FALL" if not out_after_fall else "RISE"

                inv_marker = "~" if is_active_inverted else ""
                transitions.append(TransitionTest(
                    name=f"{active_input}: 1→0{inv_marker}, others={enable_state_logical}(log) → {output}: {int(out_before_fall)}→{int(out_after_fall)} ({metric})",
                    input_signals={active_input: "1→0", **other_inputs},
                    measures=[
                        f".meas tran {metric} TRIG v({active_input.lower()}) VAL='{{{{SUPPLY/2}}}}' FALL=1 "
                        f"TARG v({output.lower()}) VAL='{{{{SUPPLY/2}}}}' {targ_edge}=1"
                    ]
                ))

        return transitions
