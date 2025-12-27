import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Literal, Union
from enum import Enum


# ============================================================
# CONFIGURATION
# ============================================================

class TestMode(Enum):
    """Mode de test pour la génération de netlist"""
    TRANSITION = "transition"  # Tests de transition pour chaque entrée
    CHARACTERIZATION = "characterization"  # Caractérisation complète (liberty)

@dataclass
class SimulationConfig:
    """Configuration générale de simulation"""
    vdd: float = 1.8
    temp: float = 27
    cload: float = 10e-15
    trise: float = 100e-12
    tfall: float = 100e-12
    test_duration: float = 2e-9
    settling_time: float = 1e-9
    corner: str = "tt"
    tran_step: str = "10p"


@dataclass
class CharacterizationConfig(SimulationConfig):
    """Configuration étendue pour la caractérisation Liberty"""
    input_loads: List[float] = None  # Capacités d'entrée à tester
    output_loads: List[float] = None  # Capacités de sortie à tester
    slew_rates: List[float] = None  # Slew rates à tester
    
    def __post_init__(self):
        if self.input_loads is None:
            self.input_loads = [1e-15, 5e-15, 10e-15]
        if self.output_loads is None:
            self.output_loads = [1e-15, 5e-15, 10e-15, 50e-15, 100e-15]
        if self.slew_rates is None:
            self.slew_rates = [50e-12, 100e-12, 200e-12, 500e-12]


@dataclass
class GateLogic:
    """Définition de la logique d'une porte"""
    function: Callable
    transition_states: Dict[str, str]


@dataclass
class TransitionTest:
    """Définition d'un test de transition"""
    name: str
    input_signals: Dict[str, str]
    measures: List[str]
    trig_pin: str = None


@dataclass
class CharacterizationTest:
    """Définition d'un test de caractérisation Liberty"""
    trig_pin: str           # Pin d'entrée (ex: 'A')
    trig_edge: str          # 'rise' ou 'fall'
    output_edge: str        # 'rise' ou 'fall' attendu en sortie
    slew_idx: int           # Index du slew rate
    load_idx: int           # Index de la charge
    t_start: float          # Temps de début (secondes)
    t_end: float            # Temps de fin (secondes)
    input_slew: float       # Slew rate d'entrée (pour PWL)



# ============================================================
# NETLIST GENERATOR CLASS
# ============================================================

class NetlistGenerator:
    """
    Générateur universel de netlists SPICE pour caractérisation de cellules standard.
    Supporte les modes transition et caractérisation complète.
    """

    def __init__(self, pdk_manager, output_dir: Optional[Path] = None, verbose: bool = False):
        self.pdk = pdk_manager
        self.verbose = verbose

        if output_dir is None:
            self.output_dir = self.pdk.pdk_root / "libs.tech" / "ngspice"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lib_spice = (
            self.pdk.pdk_root
            / "libs.ref"
            / "sky130_fd_sc_hd"
            / "spice"
            / "sky130_fd_sc_hd.spice"
        )

        # Définition logique des portes
        self.gate_logic = {
            'inv': GateLogic(lambda a: not a, {'enable': {}}),
            'buf': GateLogic(lambda a: a, {'enable': {}}),
            'nand': GateLogic(lambda *x: not all(x), {'enable': {'others': '1'}}),
            'nor': GateLogic(lambda *x: not any(x), {'enable': {'others': '0'}}),
            'and': GateLogic(lambda *x: all(x), {'enable': {'others': '1'}}),
            'or': GateLogic(lambda *x: any(x), {'enable': {'others': '0'}}),
            'xor': GateLogic(lambda *x: sum(x) % 2 == 1, {'enable': {'others': '0'}}),
            'xnor': GateLogic(lambda *x: sum(x) % 2 == 0, {'enable': {'others': '0'}}),
        }

    # ============================================================
    # HELPERS - EXTRACTION CELLULE
    # ============================================================

    def _get_cell_ports(self, cell_name: str) -> Dict:
        """
        Détecte les pins d'une cellule depuis le fichier SPICE.
        
        Returns:
            Dict avec 'input_list' et 'output'
        """
        with open(self.lib_spice, "r") as f:
            for line in f:
                ls = line.strip()
                if ls.lower().startswith(".subckt") and cell_name in ls:
                    parts = ls.split()
                    ports = parts[2:]  # skip .subckt + name
                    
                    # Convention Sky130 : sorties = Y, X, Q
                    input_list = [p for p in ports if p not in ("Y", "X", "Q", "VPWR", "VGND", "VPB", "VNB")]
                    output = next((p for p in ports if p in ("Y", "X", "Q")), None)
                    
                    return {"input_list": input_list, "output": output}
        
        raise ValueError(f"Cell {cell_name} not found in {self.lib_spice}")

    def _get_base_gate_type(self, cell_name: str) -> str:
        """
        Extrait le type de porte de base du nom de cellule.
        
        Examples:
            sky130_fd_sc_hd__nand3_1 → 'nand'
            sky130_fd_sc_hd__or4_1   → 'or'
        """
        m = re.search(r"__(\w+?)[0-9_]*$", cell_name)
        if not m:
            return "unknown"
        
        name = m.group(1)
        for gate_type in self.gate_logic.keys():
            if name.startswith(gate_type) or name == gate_type:
                return gate_type
        
        return "unknown"

    def _extract_transistors_from_cell(self, cell_name: str) -> List[str]:
        """
        Extrait les lignes de transistors d'une cellule.
        Gère les lignes de continuation avec '+'
        """
        inside = False
        devices = []
        current = ""

        with open(self.lib_spice, "r") as f:
            for line in f:
                ls = line.rstrip()

                if ls.lower().startswith(".subckt") and cell_name in ls:
                    inside = True
                    continue

                if inside and ls.lower().startswith(".ends"):
                    if current:
                        devices.append(current)
                    break

                if inside:
                    if ls.startswith("+"):
                        current += " " + ls[1:].strip()
                    else:
                        if current:
                            devices.append(current)
                        current = ls

        return devices

    # ============================================================
    # HELPERS - MODIFICATION TRANSISTORS
    # ============================================================

    def apply_transistor_widths(self, transistor_lines: List[str], w_map: Dict[str, float]) -> List[str]:
        """
        Applique les largeurs de transistors personnalisées.
        
        Args:
            transistor_lines: Lignes SPICE des transistors
            w_map: Dict {nom_transistor: largeur} ex: {'X0': 1.5e-6}
        """
        new_lines = []

        for line in transistor_lines:
            m = re.match(r"(X\S+)\s", line)
            if not m:
                new_lines.append(line)
                continue

            name = m.group(1)
            if name not in w_map:
                new_lines.append(line)
                continue

            w_new = w_map[name]
            updated = re.sub(r"w\s*=\s*[\d\.eE\-\+]+", f"w={w_new}", line)
            new_lines.append(updated)

        return new_lines

    # ============================================================
    # GENERATION DES TESTS - MODE TRANSITION
    # ============================================================

    def _generate_transition_tests(self, cell_name: str) -> List[TransitionTest]:
        """
        Génère les tests de transition pour chaque entrée.
        Teste rise et fall pour chaque pin en maintenant les autres à leur valeur d'activation.
        """
        ports = self._get_cell_ports(cell_name)
        inputs = ports["input_list"]

        gate_type = self._get_base_gate_type(cell_name)
        gate = self.gate_logic.get(gate_type, None)

        if gate is None:
            enable_val = "0"
        else:
            enable_val = gate.transition_states.get("enable", {}).get("others", "0")

        tests = []

        for idx, pin in enumerate(inputs):
            # RISE TEST
            d_rise = {}
            for j, p in enumerate(inputs):
                d_rise[p] = "rise" if j == idx else enable_val

            tests.append(
                TransitionTest(
                    name=f"{pin}_rise",
                    input_signals=d_rise,
                    measures=["rise"],
                    trig_pin=pin
                )
            )

            # FALL TEST
            d_fall = {}
            for j, p in enumerate(inputs):
                d_fall[p] = "fall" if j == idx else enable_val

            tests.append(
                TransitionTest(
                    name=f"{pin}_fall",
                    input_signals=d_fall,
                    measures=["fall"],
                    trig_pin=pin
                )
            )

        return tests

    # ============================================================
    # GENERATION DES TESTS - MODE CHARACTERIZATION
    # ============================================================
    def _generate_characterization_tests(
        self, 
        cell_name: str, 
        config: CharacterizationConfig
    ) -> List[CharacterizationTest]:
        """
        Génère les tests de caractérisation Liberty (input × slew × load).
        """
        ports = self._get_cell_ports(cell_name)
        inputs = ports["input_list"]
        
        tests = []
        test_idx = 0
        
        # Détection inverter
        is_inverting = 'inv' in cell_name.lower()
        
        # Pour chaque combinaison slew × load × input
        for slew_idx, slew in enumerate(config.slew_rates):
            for load_idx, load in enumerate(config.output_loads):
                for input_pin in inputs:
                    
                    # Test RISE
                    t_start = test_idx * config.test_duration
                    t_end = t_start + config.test_duration
                    
                    tests.append(CharacterizationTest(
                        trig_pin=input_pin,
                        trig_edge='rise',
                        output_edge='fall' if is_inverting else 'rise',
                        slew_idx=slew_idx,
                        load_idx=load_idx,
                        t_start=t_start,
                        t_end=t_end,
                        input_slew=slew
                    ))
                    test_idx += 1
                    
                    # Test FALL
                    t_start = test_idx * config.test_duration
                    t_end = t_start + config.test_duration
                    
                    tests.append(CharacterizationTest(
                        trig_pin=input_pin,
                        trig_edge='fall',
                        output_edge='rise' if is_inverting else 'fall',
                        slew_idx=slew_idx,
                        load_idx=load_idx,
                        t_start=t_start,
                        t_end=t_end,
                        input_slew=slew
                    ))
                    test_idx += 1
        
        return tests



    # ============================================================
    # GENERATION PWL
    # ============================================================

    def _build_pwl_for_transition_tests(
        self, 
        inputs: List[str], 
        tests: List[TransitionTest], 
        config: SimulationConfig
    ) -> Dict[str, List[Tuple]]:
        """
        Construit les signaux PWL pour les tests de transition.
        """
        pwl = {pin: [] for pin in inputs}
        Tslot = config.test_duration
        Tsettle = config.settling_time

        for t_index, test in enumerate(tests):
            t0 = (Tslot + Tsettle) * t_index
            t_trans = t0 + Tslot * 0.3

            for pin in inputs:
                mode = test.input_signals[pin]

                if not pwl[pin]:
                    pwl[pin].append((0, 0))

                last_val = pwl[pin][-1][1]

                if mode == "0":
                    if last_val != 0:
                        pwl[pin].append((t0, last_val))
                        pwl[pin].append((t0 + 1e-12, 0))

                elif mode == "1":
                    if last_val != config.vdd:
                        pwl[pin].append((t0, last_val))
                        pwl[pin].append((t0 + 1e-12, config.vdd))

                elif mode == "rise":
                    pwl[pin].append((t_trans, 0))
                    pwl[pin].append((f"{t_trans}+TRISE", config.vdd))

                elif mode == "fall":
                    pwl[pin].append((t_trans, config.vdd))
                    pwl[pin].append((f"{t_trans}+TFALL", 0))

                pwl[pin].append((t0 + Tslot, pwl[pin][-1][1]))

        return pwl

    def _build_pwl_for_characterization_tests(
        self, 
        inputs: List[str], 
        tests: List[CharacterizationTest], 
        config: CharacterizationConfig
    ) -> Dict[str, List[Tuple]]:
        """
        Construit les signaux PWL pour les tests de caractérisation.
        """
        pwl = {pin: [(0, 0)] for pin in inputs}
        
        gate_type = self._get_base_gate_type(tests[0].trig_pin if tests else "")
        gate = self.gate_logic.get(gate_type, None)
        enable_val = gate.transition_states.get("enable", {}).get("others", "0") if gate else "0"
        vdd = config.vdd
        
        for test in tests:
            t0 = test.t_start
            t_trans = t0 + config.test_duration * 0.3
            
            for pin in inputs:
                last_val = pwl[pin][-1][1]
                
                if pin == test.trig_pin:
                    # Cette pin fait la transition
                    if test.trig_edge == "rise":
                        if last_val != 0:
                            pwl[pin].append((t0, 0))
                        pwl[pin].append((t_trans, 0))
                        pwl[pin].append((t_trans + test.input_slew, vdd))
                    else:  # fall
                        if last_val != vdd:
                            pwl[pin].append((t0, vdd))
                        pwl[pin].append((t_trans, vdd))
                        pwl[pin].append((t_trans + test.input_slew, 0))
                else:
                    # Pin d'enable
                    target = vdd if enable_val == "1" else 0
                    if last_val != target:
                        pwl[pin].append((t0, target))
                
                pwl[pin].append((test.t_end, pwl[pin][-1][1]))
        
        return pwl


    # ============================================================
    # GENERATION DES MESURES
    # ============================================================

    def _generate_transition_measures(
        self, 
        tests: List[TransitionTest], 
        output: str, 
        config: SimulationConfig
    ) -> List[str]:
        """Génère les mesures SPICE pour les tests de transition."""
        lines = []
        Tslot = config.test_duration
        Tsettle = config.settling_time

        for idx, test in enumerate(tests):
            t0 = idx * (Tslot + Tsettle)
            t1 = t0 + Tslot
            trig_pin = test.trig_pin

            if "rise" in test.measures:
                # Propagation delay (input rise → output fall pour INV)
                lines.append(
                    f".meas tran tplh_{test.name} TRIG v({trig_pin}) VAL={config.vdd/2} RISE=1 "
                    f"TARG v({output}) VAL={config.vdd/2} FALL=1 "  # ← FALL pour inverter
                    f"FROM {t0*1e9:.3f}n TO {t1*1e9:.3f}n"
                )
                lines.append(
                    f".meas tran slew_fall_{test.name} TRIG v({output}) VAL={config.vdd*0.9} FALL=1 "
                    f"TARG v({output}) VAL={config.vdd*0.1} FALL=1 "
                    f"FROM {t0*1e9:.3f}n TO {t1*1e9:.3f}n"
                )
                    
            if "fall" in test.measures:
                lines.append(
                    f".meas tran tphl_{test.name} TRIG v({trig_pin}) VAL={config.vdd/2} FALL=1 "
                    f"TARG v({output}) VAL={config.vdd/2} RISE=1 "  # ← RISE pour inverter
                    f"FROM {t0*1e9:.3f}n TO {t1*1e9:.3f}n"
                )
                lines.append(
                    f".meas tran slew_rise_{test.name} TRIG v({output}) VAL={config.vdd*0.1} RISE=1 "
                    f"TARG v({output}) VAL={config.vdd*0.9} RISE=1 "
                    f"FROM {t0*1e9:.3f}n TO {t1*1e9:.3f}n"
                )

            lines.append(
                f".meas tran energy_{test.name} INTEG PAR('v(VPWR)*-i(Vdd)') "
                f"FROM {t0*1e9:.3f}n TO {t1*1e9:.3f}n"
            )

        return lines

    def _generate_characterization_measures(
        self, 
        tests: List[CharacterizationTest], 
        output: str, 
        config: CharacterizationConfig
    ) -> List[str]:
        """
        Génère les mesures SPICE pour caractérisation Liberty.
        """
        lines = []
        vdd = config.vdd
        v_50 = vdd / 2
        v_20 = vdd * 0.2
        v_80 = vdd * 0.8
        
        for test in tests:
            t_start = test.t_start * 1e9  # Convertir en ns
            t_end = test.t_end * 1e9
            
            pin = test.trig_pin
            trig = test.trig_edge.upper()
            out = test.output_edge.upper()
            
            # Nom de mesure
            base = f"{pin}_{trig.lower()}_s{test.slew_idx}_l{test.load_idx}"
            
            # 1. Cell delay
            lines.append(
                f".meas tran cell_{out.lower()}_{base} "
                f"TRIG v({pin}) VAL='{v_50}' {trig}=1 "
                f"TARG v({output}) VAL='{v_50}' {out}=1 "
                f"FROM={t_start:.3f}n TO={t_end:.3f}n"
            )
            
            # 2. Output transition
            if out == "RISE":
                v_start, v_stop = v_20, v_80
            else:
                v_start, v_stop = v_80, v_20
            
            lines.append(
                f".meas tran {out.lower()}_transition_{base} "
                f"TRIG v({output}) VAL='{v_start}' {out}=1 "
                f"TARG v({output}) VAL='{v_stop}' {out}=1 "
                f"FROM={t_start:.3f}n TO={t_end:.3f}n"
            )
            
            # 3. Energy
            lines.append(
                f".meas tran energy_{base} "
                f"INTEG PAR('v(VPWR)*-i(Vdd)') "
                f"FROM={t_start:.3f}n TO={t_end:.3f}n"
            )

        return lines



    # ============================================================
    # GENERATION NETLIST COMPLETE
    # ============================================================

    def _write_netlist_header(
        self, 
        cell_name: str, 
        config: SimulationConfig
    ) -> List[str]:
        """Génère l'en-tête commune des netlists"""
        lines = []
        lines.append(f"* Auto-generated testbench for {cell_name}")
        lines.append(f".lib {self.pdk.pdk_root}/libs.tech/ngspice/sky130.lib.spice {config.corner}")
        lines.append(f".temp {config.temp}")
        lines.append(f".param SUPPLY={config.vdd}")
        lines.append(f".param TRISE={config.trise}")
        lines.append(f".param TFALL={config.tfall}")
        lines.append("")
        return lines

    def _write_power_supplies(self, config: SimulationConfig) -> List[str]:
        """Génère les alimentations"""
        lines = []
        lines.append("* Power supplies")
        lines.append("Vdd VPWR 0 SUPPLY")
        lines.append("Vss VGND 0 0")
        lines.append("Vpb VPB 0 SUPPLY")
        lines.append("Vnb VNB 0 0")
        lines.append("")
        return lines

    def _write_input_sources(
        self, 
        inputs: List[str], 
        pwl_map: Dict[str, List[Tuple]], 
        config: SimulationConfig
    ) -> List[str]:
        """Génère les sources d'entrée PWL"""
        lines = []
        lines.append("* Input sources")
        
        for pin in inputs:
            pwl_s = " ".join(
                f"({t}) {v}" if isinstance(t, str) else f"{t*1e9:.4f}n {v}"
                for t, v in pwl_map[pin]
            )
            lines.append(f"V{pin} {pin} 0 PWL({pwl_s})")
        
        lines.append("")
        return lines

    def _write_dut(self, transistor_lines: List[str]) -> List[str]:
        """Génère l'instanciation du DUT"""
        lines = []
        lines.append("* Device Under Test")
        for tr in transistor_lines:
            lines.append(tr)
        lines.append("")
        return lines

    def _write_output_load(self, output: str, load: float) -> List[str]:
        """Génère la charge de sortie"""
        lines = []
        lines.append("* Output load")
        lines.append(f"CL {output} 0 {load}")
        lines.append("")
        return lines

    def _write_global_measures(self, total_time: float) -> List[str]:
        """Génère les mesures globales"""
        lines = []
        lines.append("* Global measurements")
        lines.append(
            f".meas tran energy_total INTEG PAR('v(VPWR)*-i(Vdd)') "
            f"FROM 0n TO {total_time*1e9:.3f}n"
        )
        lines.append(
            f".meas tran power_avg PARAM='energy_total/{total_time}'"
        )
        lines.append("")
        return lines

    def _write_control_section(self, total_time: float, config: SimulationConfig) -> List[str]:
        """Génère la section de contrôle"""
        lines = []
        lines.append(f".tran {config.tran_step} {total_time*1e9:.3f}n")
        lines.append(".end")
        return lines

    # ============================================================
    # FONCTIONS PUBLIQUES PRINCIPALES
    # ============================================================


    def generate_netlist(
        self,
        cell_name: str,
        mode: TestMode = TestMode.TRANSITION,
        config: Optional[Union[SimulationConfig, CharacterizationConfig]] = None,
        w_overrides: Optional[Dict[str, float]] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Génère une netlist SPICE (mode transition OU caractérisation).
        
        Args:
            cell_name: Nom de la cellule
            mode: TestMode.TRANSITION ou TestMode.CHARACTERIZATION
            config: Configuration (SimulationConfig ou CharacterizationConfig selon mode)
            w_overrides: Largeurs de transistors personnalisées
            output_path: Chemin de sortie (défaut: auto)
            
        Returns:
            Path du fichier netlist généré
        """
        # Configuration par défaut selon le mode
        if config is None:
            config = CharacterizationConfig() if mode == TestMode.CHARACTERIZATION else SimulationConfig()
        
        # Conversion Path si nécessaire
        if output_path is not None and not isinstance(output_path, Path):
            output_path = Path(output_path)

        if self.verbose:
            mode_str = "characterization" if mode == TestMode.CHARACTERIZATION else "transition"
            print(f"Generating {mode_str} netlist for {cell_name}...")

        # Extraction des informations de la cellule
        ports = self._get_cell_ports(cell_name)
        inputs = ports["input_list"]
        output = ports["output"]

        # Génération des tests selon le mode
        if mode == TestMode.CHARACTERIZATION:
            if not isinstance(config, CharacterizationConfig):
                config = CharacterizationConfig(**vars(config))
            
            tests = self._generate_characterization_tests(cell_name, config)
            pwl_map = self._build_pwl_for_characterization_tests(inputs, tests, config)
            measures_fn = self._generate_characterization_measures
            default_filename = f"{cell_name}_char.spice"
        else:
            tests = self._generate_transition_tests(cell_name)
            pwl_map = self._build_pwl_for_transition_tests(inputs, tests, config)
            measures_fn = self._generate_transition_measures
            default_filename = f"{cell_name}_tb.spice"

        # Extraction et modification des transistors
        transistors = self._extract_transistors_from_cell(cell_name)
        if w_overrides:
            transistors = self.apply_transistor_widths(transistors, w_overrides)

        # Calcul du temps total
        total_time = len(tests) * (config.test_duration + config.settling_time)

        # Assemblage de la netlist
        lines = []
        lines.extend(self._write_netlist_header(cell_name, config))
        lines.extend(self._write_power_supplies(config))
        lines.extend(self._write_input_sources(inputs, pwl_map, config))
        lines.extend(self._write_dut(transistors))
        
        # Charge de sortie (moyenne pour characterization, fixe pour transition)
        if mode == TestMode.CHARACTERIZATION:
            avg_load = sum(config.output_loads) / len(config.output_loads)
            lines.extend(self._write_output_load(output, avg_load))
        else:
            lines.extend(self._write_output_load(output, config.cload))
        
        lines.append("* Measurements")
        lines.extend(measures_fn(tests, output, config))
        lines.extend(self._write_global_measures(total_time))
        
        lines.extend(self._write_control_section(total_time, config))

        # Écriture du fichier
        if output_path is None:
            output_path = self.output_dir / default_filename
        
        output_path.write_text("\n".join(lines))

        if self.verbose:
            print(f"✓ Netlist written to {output_path}")
            print(f"  - {len(tests)} tests")
            if mode == TestMode.CHARACTERIZATION:
                print(f"  - {len(inputs)} inputs × {len(config.slew_rates)} slews × {len(config.output_loads)} loads")
            print(f"  - Total simulation time: {total_time*1e9:.2f}ns")

        return output_path



    # ============================================================
    # UTILITAIRES
    # ============================================================

    def batch_generate_netlists(
        self,
        cell_names: List[str],
        mode: TestMode = TestMode.TRANSITION,
        config: Optional[Union[SimulationConfig, CharacterizationConfig]] = None,
        w_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, Optional[Path]]:
        """
        Génère des netlists pour plusieurs cellules.
        """
        results = {}
        
        for cell_name in cell_names:
            try:
                netlist_path = self.generate_netlist(
                    cell_name=cell_name,
                    mode=mode,
                    config=config,
                    w_overrides=w_overrides
                )
                results[cell_name] = netlist_path
                
            except Exception as e:
                print(f"✗ Error processing {cell_name}: {e}")
                results[cell_name] = None
        
        return results

    def get_cell_info(self, cell_name: str) -> Dict:
        """
        Retourne les informations d'une cellule.
        
        Returns:
            Dict avec 'inputs', 'output', 'gate_type', 'transistor_count'
        """
        ports = self._get_cell_ports(cell_name)
        transistors = self._extract_transistors_from_cell(cell_name)
        gate_type = self._get_base_gate_type(cell_name)
        
        return {
            'inputs': ports['input_list'],
            'output': ports['output'],
            'gate_type': gate_type,
            'transistor_count': len(transistors),
            'transistor_names': [re.match(r"(X\S+)", t).group(1) for t in transistors if re.match(r"(X\S+)", t)]
        }

    def get_transistor_widths(self, cell_name: str) -> Dict[str, float]:
        """
        Extrait les largeurs originales des transistors d'une cellule.
        
        Args:
            cell_name: Nom de la cellule
            
        Returns:
            Dict {nom_transistor: largeur_en_metres}
            Exemple: {'X0': 4.2e-7, 'X1': 6.5e-7}
        """
        transistors = self._extract_transistors_from_cell(cell_name)
        widths = {}
        
        for line in transistors:
            # Extraire le nom du transistor
            m_name = re.match(r"(X\S+)\s", line)
            if not m_name:
                continue
            
            name = m_name.group(1)
            
            # Extraire la largeur (w=...)
            m_width = re.search(r"w\s*=\s*([\d\.eE\-\+]+)", line, re.IGNORECASE)
            if m_width:
                try:
                    width = float(m_width.group(1))
                    widths[name] = width
                except ValueError:
                    if self.verbose:
                        print(f"⚠️  Cannot parse width for {name}: {m_width.group(1)}")
            else:
                if self.verbose:
                    print(f"⚠️  No width found for {name}")
        
        return widths

# ============================================================
# EXEMPLE D'UTILISATION
# ============================================================

if __name__ == "__main__":
    from pdk_manager import PDKManager  # Votre classe existante
    
    # Initialisation
    pdk = PDKManager()
    gen = NetlistGenerator(pdk, verbose=True)
    
    # Test simple
    cell = "sky130_fd_sc_hd__nand2_1"
    
    # 1. Netlist de transition
    netlist1 = gen.generate_netlist(cell)
    
    # 2. Netlist de caractérisation
    char_config = CharacterizationConfig(
        vdd=1.8,
        slew_rates=[50e-12, 100e-12, 200e-12],
        output_loads=[10e-15, 50e-15, 100e-15]
    )
    netlist2 = gen.generate_netlist(cell, config=char_config)
    
    # 3. Batch pour plusieurs cellules
    cells = [
        "sky130_fd_sc_hd__nand2_1",
        "sky130_fd_sc_hd__nor2_1",
        "sky130_fd_sc_hd__inv_1"
    ]
    results = gen.batch_generate_netlists(cells, mode=TestMode.TRANSITION)
    
    # 4. Info sur une cellule
    info = gen.get_cell_info(cell)
    print(f"\nCell info: {info}")


