import glob
import os
import warnings
from io import StringIO
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.espresso import Espresso, EspressoProfile
from ase.data import atomic_numbers
from ase.filters import ExpCellFilter, FrechetCellFilter, UnitCellFilter  # noqa: F401
from ase.formula import Formula
from ase.io import read, write
from ase.optimize import BFGS, FIRE, FIRE2, LBFGS, BFGSLineSearch, GPMin, MDMin
from loguru import logger
from mace.calculators import MACECalculator
from scipy.spatial import ConvexHull

from mc_mace.utils.io import append_line_to_file, clean_ase_read, save_dict_to_yaml
from mc_mace.utils.parse import parse_yaml_voltage_input
from mc_mace.utils.profiler import MethodProfiler

from .simulation_abc import BaseSimulation

profiler_io = MethodProfiler(name="Profiling I/O")
profiler_calc = MethodProfiler(name="Profiling Calculation")


class VoltageCalculator:
    def __init__(self, states_files, working_ion, working_ion_energy, charge_carried):
        self.charge_carried = charge_carried
        self.states_files = states_files
        self.working_ion = working_ion
        self.working_ion_energy = working_ion_energy
        self._state_energy = {}
        self._state_formation_energy = []
        self._energy_full = 0
        self._formula_full = ""
        self._energy_empty = 0
        self._formula_empty = ""
        self._n_ion_max = 0
        self.reduce_factor = 0
        self._convexhull = None
        self._stable_points = None
        self.voltage_steps = []
        self.number_of_ions = 0  # Number of ions in the unit formula

    def get_state_energy(self):
        for file_path in self.states_files:
            formula = os.path.basename(file_path)[4:-4]
            data = pd.read_csv(file_path)
            self._state_energy[formula] = data["energy"].values

    def get_extremes(self):
        """
        Identify the states with the maximum and minimum number of ions.

        This method iterates through the energy states and determines:
        - The state with the maximum number of working ions (`_formula_full`).
        - The state with the minimum number of working ions (`_formula_empty`).

        Raises:
            ValueError: If multiple energy values are found for the same state.

        Attributes Updated:
            - _formula_full: The formula of the state with the maximum working ions.
            - _formula_empty: The formula of the state with the minimum working ions.
            - _energy_full: Energy of the state with the maximum working ions.
            - _energy_empty: Energy of the state with the minimum working ions.
        """
        if len(self._state_energy[self._formula_full]) > 1:
            print("Too many energy in full state")
            print(self._state_energy[self._formula_full])
        self._energy_full = self._state_energy[self._formula_full]
        if len(self._state_energy[self._formula_empty]) > 1:
            print(self._state_energy[self._formula_empty])
            print("Too many energy in empty state")
        self._energy_empty = self._state_energy[self._formula_empty]

    def get_n_max(self):
        """
        Attributes Updated:
            - _n_ion_max: Maximum number of working ions in any state.
        """
        n_min = float("inf")
        n_max = 0
        for formula_, energy in self._state_energy.items():
            element_count = Formula(formula_).count()
            if self.working_ion in element_count:
                n_ion = element_count[self.working_ion]
            else:
                n_ion = 0
            if n_ion > n_max:
                self._formula_full = formula_
                n_max = n_ion
            if n_ion < n_min:
                self._formula_empty = formula_
                n_min = n_ion
        logger.debug(
            f"Full state: {self._formula_full} with # ions: {n_max}, empty state: {self._formula_empty} with # ions: {n_min}, working ion: {self.working_ion} with energy {self.working_ion_energy}"
        )
        self._n_ion_max = n_max

    def get_reduce_factor(self):
        self.reduce_factor = Formula(self._formula_full).reduce()[1]

    def get_formation_energy(self):
        for formula_, energy in self._state_energy.items():
            element_count = Formula(formula_).count()
            if self.working_ion in element_count:
                n_ion = element_count[self.working_ion]
            else:
                n_ion = 0
            x = n_ion / self._n_ion_max
            e_extreme = (self._n_ion_max - n_ion) / self._n_ion_max * self._energy_empty + n_ion / self._n_ion_max * (
                self._energy_full
            )
            self._state_formation_energy.append((formula_, n_ion, x, (energy - e_extreme) / self.reduce_factor))

    def get_convexhull(self):
        points = np.array([[x, ef] for _, _, x, efs in self._state_formation_energy for ef in efs])
        self._convexhull = ConvexHull(points)
        hull_points = points[self._convexhull.vertices]
        hull_points = hull_points[np.argsort(hull_points[:, 0])]
        self._stable_points = hull_points

    def get_number_of_ions(self):
        self.get_state_energy()
        self.get_n_max()
        self.get_reduce_factor()
        self.number_of_ions = self._n_ion_max / self.reduce_factor

    def get_voltage(self):
        self.voltage_steps = np.zeros((self._stable_points.shape[0] - 1, 3))
        e_full_per_formula = self._energy_full / self.reduce_factor
        e_empty_per_formula = self._energy_empty / self.reduce_factor
        for i in range(1, self._stable_points.shape[0]):
            x1, e1 = self._stable_points[i - 1]
            x2, e2 = self._stable_points[i]
            delta_e = (
                e2
                - e1
                + (x2 - x1)
                * (e_full_per_formula - e_empty_per_formula - (self.working_ion_energy * self.number_of_ions))
            )
            delta_x = x2 - x1
            voltage = -delta_e / (delta_x * self.number_of_ions * self.charge_carried)  # Voltage in volts
            # self.voltage_steps.append([float(x1), float(x2), float(voltage)])
            self.voltage_steps[i - 1, :] = [
                x1,
                x2,
                voltage[0],
            ]  # [x1, x2, voltage] #[float(x1), float(x2), float(voltage)]
            logger.debug(
                f"Stable points: x1 = {x1}, x2 = {x2}, delta_x = {delta_x}, e1 = {e1}, e2 = {e2}, delta_e =  {delta_e}, voltage = {voltage[0]}"
            )

    def write_voltage(self, file_path):
        with open(file_path, "w") as f:
            f.write("x1,x2,V\n")
            for x1, x2, voltage in self.voltage_steps:
                f.write(f"{x1:12.8f},{x2:12.8f},{voltage:12.8f}\n")

    def write_convexhull(self, file_path):
        with open(file_path, "w") as f:
            f.write("frmula,n ions,x,formation energy\n")
            for fromula, n_ion, x, efs in self._state_formation_energy:
                for ef in efs:
                    f.write(f"{fromula:12s},{n_ion:12d},{x:12.8f},{ef:12.8f}\n")


class VoltageProfile(BaseSimulation):
    """
    Subclass to compute the voltage profile of a system at 0K using MACE force fields or Quantum espresso DFT.

    This subclass removes atoms iteratively and optimizes the structure to compute energy changes.
    """

    def __init__(self, input_file: Path | str, log_file: Path | str, log_level: str, colorize: bool, device: str):
        super().__init__(input_file, log_file, log_level, colorize, device)
        self._optimizer_type: str = "FIRE2"
        self._max_steps: int = 1000
        self._fmax: float = 0.05
        self.state_0: Atoms
        self.state_1: Atoms
        self._i_state: int  # current state
        self._ai: list  # removed atom id
        self.n_states: int  # number of states
        self._converged: bool
        self.save_trj_step: int
        self.save_thermo_step: int
        self.save_state_step: int
        # OUT
        self._energy_store_thermo: list = []
        self.out_thermo: str | Path
        self.out_trj: str | Path
        self.out_state_folder: str | Path
        self.out_voltage: str | Path
        self.out_convexhull: str | Path
        self.save_trj_step: int = 1
        self.save_thermo_step: int = 1
        self.save_state_step: int = 1
        self.saved_state_files: list[str | Path] = []
        # Voltage
        self._voltage_calculator: VoltageProfile
        self._charge_carried: float

    def __logger_prefix(self) -> str:
        try:
            prefix = "[" + f"State {self._i_state:d}/{self.n_states -1:d}".center(19, " ") + "] "
        except AttributeError:
            prefix = ""
        return prefix

    # Turn off Abstract methods
    def _set_engine(self) -> None:
        pass

    def _get_ensemble_settings(self) -> None:
        pass

    # Add new methods
    def _pw_input_file(self):
        """
        Extract quantum espresso input parameters to be provided to the espresso calculator.
        """
        pw_input = {
            "calculation": self.sim_settings["calculation"],
            "restart_mode": self.sim_settings["restart_mode"],
            "verbosity": self.sim_settings["verbosity"],
            "outdir": self.sim_settings["outdir"],
            "prefix": self.sim_settings["prefix"],
            "max_seconds": self.sim_settings["max_seconds"],
            "tstress": self.sim_settings["tstress"],
            "tprnfor": self.sim_settings["tprnfor"],
            "nstep": self.sim_settings["nstep"],
            "etot_conv_thr": self.sim_settings["etot_conv_thr"],
            "forc_conv_thr": self.sim_settings["forc_conv_thr"],
            "input_dft": self.sim_settings["input_dft"],
            "occupations": self.sim_settings["occupations"],
            "degauss": self.sim_settings["degauss"],
            "smearing": self.sim_settings["smearing"],
            "conv_thr": self.sim_settings["conv_thr"],
            "electron_maxstep": self.sim_settings["electron_maxstep"],
            "mixing_mode": self.sim_settings["mixing_mode"],
            "mixing_beta": self.sim_settings["mixing_beta"],
            "diagonalization": self.sim_settings["diagonalization"],
            "startingwfc": self.sim_settings["startingwfc"],
            "ecutwfc": self.sim_settings["ecutwfc"],
            "ecutrho": self.sim_settings["ecutrho"],
            "nspin": self.sim_settings["nspin"],
            "starting_magnetization(1)": self.sim_settings["starting_magnetization(1)"],
            "starting_magnetization(2)": self.sim_settings["starting_magnetization(2)"],
            "starting_magnetization(3)": self.sim_settings["starting_magnetization(3)"],
            "starting_magnetization(4)": self.sim_settings["starting_magnetization(4)"],
            "starting_magnetization(5)": self.sim_settings["starting_magnetization(5)"],
            "starting_magnetization(6)": self.sim_settings["starting_magnetization(6)"],
        }
        return pw_input

    def _get_profile(self) -> EspressoProfile:
        """
        Specify the path to the pw.x executable and the pseudopotential directory.
        """
        command = self.sim_settings["command"]
        pseudo_dir = self.sim_settings["pseudo_dir"]

        return EspressoProfile(
            command=command,
            pseudo_dir=pseudo_dir,
        )

    def read_restart_files(self):
        """
        Read the csv and xyz file for restart
        """
        file_pattern = "[0-9][0-9][0-9]-*"
        files = glob.glob(file_pattern + ".xyz", root_dir=self.out_state_folder)
        csv_files = glob.glob(f"{self.out_state_folder}/{file_pattern}.csv")

        return files, csv_files

    def delete_csv(
        self,
    ) -> None:
        """
        Delete the csv file for single step restart.
        """
        _, csv_files = self.read_restart_files()
        file = glob.glob(f"{self.out_state_folder}/{self._i_state:03d}-*.csv")
        if len(file) != 0:
            logger.debug(f"Deleting csv file {file[0]}")
            os.remove(file[0])

    def _restart(self) -> None:
        """
        Continue the simulation from the previous state.
        """

        files, csv_files = self.read_restart_files()

        if len(csv_files) != len(files):
            raise RuntimeError(
                f"The number of .csv files should be the same as the number of .xyz files. Check {self.out_state_folder} or choose continue: False to start from scratch."
            )
        if not files:
            logger.info("No files found for restarting. Volta profile will start from scratch.")
            self._scratch()
        else:
            self.saved_state_files.extend(csv_files)
            if len(files) == self.n_states:
                #raise RuntimeError("Last state found, no more states to compute.")
                logger.warning("Last state found, no more states to compute.")
                self._custom_convex_hull()
                self.post_process()

            elif len(files) == 1:  # in automatic restart, this state must be the initial state (fully intercalated)
                logger.info("Continue: optimizing the final state")
                # _file = files[0]

                self.optimize_last_configuration()
                self.check_ave_voltage()

                self.state_0 = self.system.copy()
                cont_state = 0
            elif (
                len(files) == 2
            ):  # in automatic restart, those 2 states must be the initial (fully intercalated) and final states (fully de-intercalated)
                logger.info("Continue: initial and final state found, starting the de-intercalation process")
                self.state_0 = self.system.copy()
                cont_state = 0
            else:  # continue from the last attempt
                # _max_xyz_file = max(files, key=lambda x: int(x[:3]))
                sorted_files = sorted(files, key=lambda x: int(x[:3]))
                _max_xyz_file = sorted_files[-2]
                # _max_csv_file = max(csv_files, key=lambda x: int(x.split("/")[-1].split("-")[0]))
                sorted_csv_files = sorted(csv_files, key=lambda x: int(x.split("/")[-1].split("-")[0]))
                _max_csv_file = sorted_csv_files[-2]

                self._i_state = int(_max_xyz_file[:3])

                # files.remove(_max_xyz_file)
                # _file = max(files, key=lambda x: int(x[:3]))
                _file = sorted_files[-3]
                logger.info(
                    "Continue: continue de-intercalation process of step " + str(self._i_state),
                    "starting from state " + _file,
                )

                self._continue_interrupted_step(_file, _max_csv_file)
                self.post_process()
                self.state_0 = read(self.out_state_folder + "/" + _max_xyz_file)
                cont_state = int(_max_xyz_file[:3])

            # logger.debug(f"Continuing volta profile from {_max_xyz_file}")

            for self._i_state in range(cont_state + 1, self.n_states - 1):
                _energy = self._find_atom_to_remove()

                self.update_files(_energy)
                self.update_states()
                self.post_process()

    def _continue_interrupted_step(self, xyz_file, csv_file) -> None:
        """
        restart the last interrupted step
        """
        data = pd.read_csv(csv_file, header=None, skiprows=1)

        if self.sim_settings["removal_method"] == "brute_force":
            # Read atom indices that have been already removed
            # Use tuples for faster comparison between done vs to be done combinations
            id_done = {tuple(map(int, comb)) for comb in data.iloc[:, 0 : self._i_state].values}
            logger.debug(f"Atoms done: {id_done}")
            system = self.system.copy()
            num_Li_to_be_removed = self._i_state
            atom_indices = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]

            # Generate all combinations of atom indices to remove
            # And check if the combination has already been done
            for combo in combinations(atom_indices, self._i_state):
                if tuple(map(int, combo)) not in id_done:  # Compare as tuples of ints
                    self._remove(system, self._i_state, list(combo))

        elif self.sim_settings["removal_method"] == "semi_brute_force":
            id_done = data.iloc[:, 0].values.astype(int)
            logger.debug(f"Atoms done: {id_done}")
            system = read(self.out_state_folder + "/" + xyz_file)
            num_Li_to_be_removed = 1
            atom_to_remove = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]
            atom_to_remove = np.setdiff1d(atom_to_remove, id_done)  # remove done atoms
            if len(atom_to_remove) == 0:
                logger.warning("No atoms to remove, moving to the next step.")
                return
            else:
                self._remove(system, num_Li_to_be_removed, atom_to_remove)
        else:
            logger.error(f"Unsupported removal method {self.sim_settings['removal_method']}")
            raise ValueError(f"Unsupported removal method {self.sim_settings['removal_method']}")

    def _scratch(self) -> None:
        """
        Start the simulation from scratch.
        """
        for self._i_state in range(self.n_states - 1):  # self.n_states -1
            logger.info(f" State {self._i_state} ".center(120, "-"))
            if self._i_state == 0:

                self.optimize_initial_configuration()
                # self.update_files(energy_end)
                energy_end = self.optimize_last_configuration()

                self.update_files(energy_end)
                self.check_ave_voltage()
                self.update_states()

            else:
                energy_end = self._find_atom_to_remove()
                self.post_process()

                self.update_files(energy_end)
                self.update_states()

    def optimize_initial_configuration(self) -> None:
        """
        Optimize fully intercalated system
        """
        logger.info("Optimizing initial (fully intercalated) configuration")
        self._ai = [-1]
        self.state_0 = self.system.copy()
        self.state_1 = self.state_0.copy()
        logger.info(
            self.__logger_prefix()
            + f"Optimizing (position and cell) initial configuration {self.state_0.get_chemical_formula()}"
        )
        self._set_calculator(str(self._ai) + "_" + self.state_1.get_chemical_formula())
        self.state_1.calc = self.calculator
        energy_start = self._get_potential_energy_new_state()
        if self.sim_settings.get("calculation") == "vc-relax":
            # Check if vc-relaxation is finished
            tag = ""
            for val in self._ai:
                tag += str(val) + "_"
        force_max_start = np.max(np.linalg.norm(self.state_1.get_forces(), axis=1))
        cell_start = self.state_1.cell.cellpar()
        vol_start = self.state_1.cell.volume
        logger.debug(f"optimizing system {self.state_1.get_chemical_formula()}")
        self.state_1, energy_end = self._optimize_system(self.state_1)
        force_max_end = np.max(np.linalg.norm(self.state_1.get_forces(), axis=1))
        cell_end = self.state_1.cell.cellpar()
        vol_end = self.state_1.cell.volume
        logger.info(self.__logger_prefix() + f"Energy {energy_start:.3f} eV -> {energy_end:.3f} eV")
        logger.info(self.__logger_prefix() + f"Max Force {force_max_start:.3e} eV/A -> {force_max_end:.3e} eV/A")
        logger.info(self.__logger_prefix() + f"Cell lengths [a, b, c] {cell_start[:3]} A -> {cell_end[:3]} A")
        logger.info(self.__logger_prefix() + f"Cell angles [α,β,γ] {cell_start[3:]} ° -> {cell_end[3:]} °")
        logger.info(self.__logger_prefix() + f"Cell volume {vol_start:.3f} A^3 -> {vol_end:.3f} A^3")
        # self._energy_store_thermo.append(self._get_potential_energy_new_state())
        self._energy_store_thermo.append(energy_end)
        self.save_state(energy_end)
        # self._ai = -1
        # self.state_1 = self.system.copy()
        # opt_state = self._optimize_system(self.state_1.copy())
        # self.state_1 = opt_state.copy()
        # self.state_1.calc = self.calculator
        # self._energy_store_thermo.append(self.state_1.get_potential_energy())
        # self.save_state()
        # self.state_0 = opt_state.copy()

    def optimize_last_configuration(self) -> float:
        """
        Optimize fully de-intercalated system
        """
        logger.info("Optimizing final (fully de-intercalated) configuration")
        self._i_state = self.n_states - 1
        self.state_1 = self.system.copy()
        num_Li_to_be_removed = self.n_states - 1
        atom_to_remove = np.where(self.state_1.get_atomic_numbers() == atomic_numbers[self.element])[0]
        deintercalated_energy = self._remove(
            self.state_1, num_Li_to_be_removed, atom_to_remove, final=True
        )  # if I use this, in the csv file the format will not work in the semi brute force method
        self.state_1 = self.system.copy()  # to give correct state_1 for the next step
        return deintercalated_energy

    def check_ave_voltage(self) -> None:
        """
        Check if the average voltage is within specified limits and stop the calculation if not.

        This method is called after optimizing the fully intercalated and de-intercalated states.
        It computes the average voltage using only the energies of these two extreme states.
        If the estimated average voltage exceeds `voltage_max` or falls below `voltage_min` (as specified in `sim_settings`),
        the calculation is stopped early by raising a ValueError.

        Raises:
            ValueError: If the average voltage is higher than `voltage_max` or lower than `voltage_min`.

        Attributes Used:
            - self.sim_settings["voltage_max"]: Maximum allowed voltage.
            - self.sim_settings["voltage_min"]: Minimum allowed voltage.
            - self._voltage._state_energy: Dictionary of state energies.
            - self._voltage.reduce_factor: Reduction factor for normalization.
            - self._voltage.number_of_ions: Number of working ions.
            - self.chemical_potential: Chemical potential of the working ion.
            - self._charge_carried: Charge carried by the working ion.
        """
        self._voltage = VoltageCalculator(
            self.saved_state_files,
            self.element,
            self.chemical_potential,
            self._charge_carried,
        )
        self._voltage.get_state_energy()
        self._voltage.get_number_of_ions()
        self._voltage.get_reduce_factor()
        intercalated_energy = self._voltage._state_energy[f"{self._voltage._formula_full}"]
        deintercalated_energy = self._voltage._state_energy[f"{self._voltage._formula_empty}"]
        ave_voltage = -(
            (intercalated_energy - deintercalated_energy) / self._voltage.reduce_factor
            - (self._voltage.number_of_ions * self.chemical_potential)
        ) / (self._voltage.number_of_ions * self._charge_carried)
        logger.info(f"Initial average voltage estimation {ave_voltage}")
        if self.sim_settings["voltage_max"] is not None and ave_voltage > self.sim_settings["voltage_max"]:
            raise ValueError(
                f"Voltage (={ave_voltage}) exceeds the maximum limit of {self.sim_settings['voltage_max']} V."
            )
        elif self.sim_settings["voltage_min"] is not None and ave_voltage < self.sim_settings["voltage_min"]:
            raise ValueError(
                f"Voltage (={ave_voltage}) subceeds the minmum limit of {self.sim_settings['voltage_min']} V."
            )

    # Change Abstract methods
    def _load_system(self) -> None:
        self.system = clean_ase_read(self.sim_settings["system"])  # type: ignore[index]

    # Change Abstract methods
    def _set_calculator(self, sub_file: str = "temp") -> None:
        if "mace_model" in self.sim_settings:
            with warnings.catch_warnings():
                logger.debug("Loading MACE model")

                self.calculator = MACECalculator(model_paths=self.sim_settings["mace_model"], device=self.device)  # type: ignore[index]

        elif "calculation" in self.sim_settings:
            logger.debug("Loading Quantum espresso calculator")
            self.calculator = Espresso(
                input_data=self._pw_input_file(),
                profile=self._get_profile(),
                pseudopotentials=self.sim_settings["pseudopotentials"],
                kpts=self.sim_settings["kpts"],
                koffset=self.sim_settings["koffset"],
                directory=self.sim_settings["QE_dir"] + "/" + sub_file,
            )

    def _compute_chemical_potentials(self) -> tuple[list[str], list[float]]:
        """
        Compute chemical potentials.

        Returns:
            tuple[list[str], list[float]]: A tuple containing a list of elements and their chemical potentials.
        """
        if self.sim_settings is None:
            logger.warning("Wrong class execution, run the `initialize` method first")
            self.initialize()
        elements = []
        potentials = []
        if len(list(self.sim_settings["working ion"]["chemical potential"].keys())) > 1:  # type: ignore[index]
            logger.error("Only one chemical potential at the time can be tuned")
            logger.error(f"Found more that one chemical potential in {self.input_file}")
            logger.error(f"chemical potential: {self.sim_settings['working ion']['chemical potential']}")  # type: ignore[index]
            raise NotImplementedError("Only one chemical potential at the time can be tuned")
        for element, value in self.sim_settings["working ion"]["chemical potential"].items():  # type: ignore[index]
            elements.append(element)
            if isinstance(value, str):
                logger.debug(f"Computing the chemical potential from energy of system {value}")
                atoms = read(value)
                self._set_calculator(element)
                atoms.calc = self.calculator
                if not np.all(atoms.get_atomic_numbers() == atoms.get_atomic_numbers()[0]):
                    logger.error(f"The system {value} contain more the one element")
                element = atoms.get_chemical_symbols()[0]
                atoms.calc = self.calculator
                atoms, energy_end = self._optimize_system(atoms)
                mu = float(energy_end / len(atoms))
                logger.info(f"mu({element}) = {mu:.3f} eV")
                self.sim_settings["working ion"]["chemical potential"][element] = mu  # type: ignore[index]
                potentials.append(mu)
            else:
                potentials.append(value)
        self.element = elements[0]
        self.chemical_potential = potentials[0]

    def _write_file_headers(self) -> None:
        """
        Write headers to output files for thermo and events data.

        This method ensures that all required output files start with the appropriate
        header information, enabling structured data logging during the simulation.
        """
        if self.out_thermo is not None:
            self._out_thermo_header()

    def update_files(self, _energy: float) -> None:
        """
        Update all output files according to the current simulation step.

        This method manages writing data to trajectory, restart, thermo, state, and events files
        based on their respective save frequencies.
        """
        if self.out_trj is not None and self._i_state % self.save_trj_step == 0:  # type: ignore[operator]
            self.save_trj(_energy)
        if self.out_thermo is not None and self._i_state % self.save_thermo_step == 0:  # type: ignore[operator]
            self.save_thermo(_energy)
        # if self.out_state_folder is not None and self._i_state % self.save_state_step == 0 or self._new_state:  # type: ignore[operator]
        #     self.save_state()

    @profiler_io.track
    def save_trj(self, _energy: float) -> None:
        """
        Save the current state of the simulation to the trajectory file.

        This method writes atomic configuration, energy, volume, and step information
        to the trajectory file at the specified save frequency.

        Raises:
            ValueError: If the trajectory file path is not set.
        """
        atoms = self.state_1.copy()
        energy = _energy
        volume = self.state_1.get_volume()
        atoms.info["volume"] = volume
        atoms.info["potential_energy"] = energy
        atoms.info["energy"] = energy
        atoms.info["state"] = self._i_state
        atoms.info["converged"] = self._converged
        write(self.out_trj, atoms, append=True)
        logger.debug(f"Updated trajectory file: {self.out_trj}")

    def _out_thermo_header(self) -> None:
        if self.out_thermo is not None:
            append_line_to_file(
                self.out_thermo,
                f"{'state'},{'energy'},{'mean energy'},{'std energy'},{'max energy'},{'min energy'},{'volume'},{'atoms'}",
            )

    @profiler_io.track
    def save_thermo(self, _energy: float) -> None:
        """
        Save thermodynamic properties to the thermo file.

        This method writes energy, volume, and atom count statistics to the thermo file
        at specified save frequencies.

        Raises:
            ValueError: If the thermo file path is not set.
        """
        if self.out_thermo is not None:
            atoms = self.state_1.copy()
            energy = _energy
            volume = self.state_1.get_volume()
            mean_energy = np.nanmean(self._energy_store_thermo)
            std_energy = np.nanstd(self._energy_store_thermo)
            max_energy = np.nanmax(self._energy_store_thermo)
            min_energy = np.nanmin(self._energy_store_thermo)
            append_line_to_file(
                self.out_thermo,
                (
                    f"{self._i_state:10d}, {energy:15.8e}, {mean_energy:15.8e}, "
                    f"{std_energy:15.8e}, {max_energy:15.8e}, {min_energy:15.8e}, "
                    f"{volume:15.8e}, {len(atoms):15d}"
                ),
            )
            logger.debug(f"Updated thermo file: {self.out_thermo}")

    def _save_state_xyz(self, file_path: str, _energy: float) -> None:
        atoms = self.state_1.copy()
        energy = _energy
        volume = self.state_1.get_volume()
        atoms.info["volume"] = volume
        atoms.info["potential_energy"] = energy
        atoms.info["energy"] = energy
        atoms.info["state"] = self._i_state
        atoms.info["removed_atom"] = self._ai
        write(file_path, atoms, append=False)
        logger.info(f"Saving state file {file_path}")

    def _state_file_header(self, file_path: str | Path) -> None:
        append_line_to_file(
            file_path,
            f"{'atoms_id'},{'energy'},{'volume'},{'atoms'}",
        )

    @profiler_io.track
    def save_state(self, _energy: float) -> None:
        """
        Save the current simulation state to a state file.

        This method writes atomic configurations, energy, and volume data to a state
        file in both `.csv` and `.xyz` formats. It ensures no duplicate state files
        for the same configuration.

        Raises:
            ValueError: If the state folder path is not set.
            RuntimeError: If multiple state files with the same formula exist in the folder.
        """
        if self.out_state_folder is not None:
            atoms = self.state_1.copy()
            energy = _energy
            volume = self.state_1.get_volume()
            formula = atoms.get_chemical_formula(mode="hill", empirical=False)
            files = glob.glob(os.path.join(str(self.out_state_folder), f"*-{formula}.csv"))
            if len(files) < 1:
                # id = len(glob.glob(os.path.join(str(self.out_state_folder), "*.csv")))
                id = self._i_state
                file_path = os.path.join(str(self.out_state_folder), f"{id:03d}-{formula}.csv")
                self.saved_state_files.append(file_path)
                self._state_file_header(file_path)
                conf_file_path = os.path.join(str(self.out_state_folder), f"{id:03d}-{formula}.xyz")
            elif len(files) == 1:
                file_path = files[-1]
                conf_file_path = files[-1][:-3] + "xyz"
            else:
                logger.critical(f"Found multiple state files with formula {formula}, in folder {self.out_state_folder}")
                raise ValueError(
                    f"Found multiple state files with formula {formula}, in folder {self.out_state_folder}"
                )

            atoms_id = ", ".join(f"{x:<10d}" for x in self._ai)
            append_line_to_file(
                file_path,
                f"{atoms_id:<10s}, {energy:15.8e}, {volume:15.8e}, {len(atoms):15d}",
            )
            self._save_state_xyz(conf_file_path, _energy)
            logger.debug(f"Updated state file: {file_path}")

    @profiler_calc.track
    def _optimize_system(self, atoms: Atoms):
        """
        Optimize the atomic system with a filter to include box relaxation.

        Args:
            atoms (Atoms): ASE Atoms object representing the system.

        Returns:
            float: Optimized potential energy of the system.
        """
        if self.sim_settings.get("calculation") == "vc-relax":
            logger.debug("Optimization method: quantum espresso vc-relaxation")
            try:
                _energy = atoms.get_potential_energy()
                self._converged = True
            except Exception as e:
                logger.error(f"Error: {e}")
                self._converged = False

        else:
            logger.debug(f"Optimization method: {self._optimizer_type} with {self.calculator}")
            atoms.calc = self.calculator
            ucf = FrechetCellFilter(atoms)  # ExpCellFilter(atoms)
            out = StringIO()
            if self._optimizer_type.upper() == "BFGS":
                optimizer = BFGS(ucf, logfile=out)
            elif self._optimizer_type.upper() == "LBFGS":
                optimizer = LBFGS(ucf, logfile=out)
            elif self._optimizer_type.upper() == "FIRE":
                optimizer = FIRE(ucf, logfile=out)
            elif self._optimizer_type.upper() == "FIRE2":
                optimizer = FIRE2(ucf, logfile=out)
            elif self._optimizer_type.upper() == "GPMIN":
                optimizer = GPMin(ucf, logfile=out)
            elif self._optimizer_type.upper() == "MDMIN":
                optimizer = MDMin(ucf, logfile=out)
            elif self._optimizer_type.upper() == "BFGSLineSearch".upper():
                optimizer = BFGSLineSearch(ucf, logfile=out)
            else:
                logger.error(f"Unsupported optimizer type {self._optimizer_type}")
                raise ValueError(f"Unsupported optimizer type {self._optimizer_type}")

            # with redirect_stdout(out):
            self._converged = optimizer.run(fmax=self._fmax, steps=self._max_steps)
            for line in out.getvalue().splitlines():
                logger.debug(self.__logger_prefix() + line)
            if not self._converged:
                raise RuntimeError(
                    f"The optimization with {self._optimizer_type} not converged after {self._max_steps} steps"
                )
            self.state_1 = atoms
            _energy = self._get_potential_energy_new_state()
        return atoms, _energy

    @profiler_calc.track
    def _get_potential_energy_new_state(self):
        logger.debug("Computing potential energy")
        return self.state_1.get_potential_energy()

    def _find_atom_to_remove(self) -> float:
        """
        Identify the atom to remove from the system.

        brute_force: Attempts are made starting from original system.
        semi_brute_force: Attempts are made from the lowest energy structure of previous step.
        genetic: Genetic algorithm, not implemented yet.
        cluster_expansion: Not implemented yet.
        """
        if self.sim_settings["removal_method"] == "brute_force":
            logger.info("Brute force method")
            system = self.system.copy()
            atom_to_remove = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]
            num_Li_to_be_removed = self.n_states - len(
                np.where(self.state_0.get_atomic_numbers() == atomic_numbers[self.element])[0]
            )
            return self._remove(system, num_Li_to_be_removed, atom_to_remove)
        elif self.sim_settings["removal_method"] == "semi_brute_force":
            logger.info("Semi brute force method")
            system = self.state_0.copy()
            atom_to_remove = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]
            num_Li_to_be_removed = 1
            return self._remove(system, num_Li_to_be_removed, atom_to_remove)
        elif self.sim_settings["removal_method"] == "genetic":
            logger.info("Genetic algorithm method")
            raise NotImplementedError("Genetic algorithm not implemented")
        elif self.sim_settings["removal_method"] == "cluster_expansion":
            logger.info("Cluster expansion method not implemented yet")
            raise NotImplementedError("Cluster expansion not implemented")

    def _remove(self, system: Atoms, num_Li_to_be_removed: int, atom_to_remove: np.array, final: bool = False) -> float:
        """
        Identify the atom to remove by computing energy differences.

        Returns:
            int: Index of the atom to remove.
            float: The minimum total energy obtained after removing one atom.
        """
        min_energy = float("inf")
        best = None
        best_ai = None
        # atom_to_remove = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]
        logger.info(
            self.__logger_prefix() + f"Removing {num_Li_to_be_removed} atom from system {system.get_chemical_formula()}"
        )
        self._energy_store_thermo = []
        for self._ai in list(combinations(atom_to_remove, num_Li_to_be_removed)):
            self.state_1 = system.copy()
            del self.state_1[list(self._ai)]
            logger.info(
                self.__logger_prefix()
                + f"Optimizing {self.state_1.get_chemical_formula()} (position and cell) after removing atom id:{self._ai} from system {system.get_chemical_formula()}"
            )
            tag = ""
            for val in self._ai:
                tag += str(val) + "_"
            self._set_calculator(f"{tag + self.state_1.get_chemical_formula()}")
            self.state_1.calc = self.calculator
            energy_start = self._get_potential_energy_new_state()
            logger.debug(f"Optimizing system {self.state_1.get_chemical_formula()}")
            force_max_start = np.max(np.linalg.norm(self.state_1.get_forces(), axis=1))
            cell_start = self.state_1.cell.cellpar()
            vol_start = self.state_1.cell.volume
            self.state_1, energy_end = self._optimize_system(self.state_1)
            force_max_end = np.max(np.linalg.norm(self.state_1.get_forces(), axis=1))
            cell_end = self.state_1.cell.cellpar()
            vol_end = self.state_1.cell.volume
            logger.info(self.__logger_prefix() + f"Energy {energy_start:.3f} eV -> {energy_end:.3f} eV")
            logger.info(self.__logger_prefix() + f"Max Force {force_max_start:.3e} eV/A -> {force_max_end:.3e} eV/A")
            logger.info(self.__logger_prefix() + f"Cell lengths [a, b, c] {cell_start[:3]} A -> {cell_end[:3]} A")
            logger.info(self.__logger_prefix() + f"Cell angles [α,β,γ] {cell_start[3:]} ° -> {cell_end[3:]} °")
            logger.info(self.__logger_prefix() + f"Cell volume {vol_start:.3f} A^3 -> {vol_end:.3f} A^3")
            if final:  # if this is the last step, use atom id -1 in the csv file
                self._ai = [-1]
            self._energy_store_thermo.append(energy_end)
            self.save_state(energy_end)
            if energy_end < min_energy:
                min_energy = energy_end
                best = self.state_1.copy()
                best_ai = self._ai
        logger.debug(
            self.__logger_prefix()
            + f"Minimal energy obtained by removing {best_ai} atom form {system.get_chemical_formula()}"
        )
        self.state_1 = best.copy()
        self.state_1.calc = self.calculator
        return min_energy

    def genetic(self):
        """
        Genetic algorithm to find the best atom to remove.
        """
        pass

    def cluster_expansion(self):
        """
        Cluster expansion to find the best atom to remove.
        """
        pass

    def update_states(self):
        self.state_0 = self.state_1.copy()

    def _load_settings(self) -> None:
        self.sim_settings = parse_yaml_voltage_input(self.input_file)

    def initialize(self) -> None:
        """
        Initialize the Voltage profile simulation.
        """
        super().initialize()
        self._optimizer_type = self.sim_settings["optimizer"]["type"]
        self._fmax = self.sim_settings["optimizer"]["fmax"]
        self._max_steps = self.sim_settings["optimizer"]["max steps"]
        self._compute_chemical_potentials()
        self.n_states = len(np.where(self.system.get_atomic_numbers() == atomic_numbers[self.element])[0]) + 1
        self.out_thermo = self.sim_settings["output files"]["thermo"]
        self.out_trj = self.sim_settings["output files"]["trajectory"]
        self.out_voltage = self.sim_settings["output files"]["voltage"]
        self.out_convexhull = self.sim_settings["output files"]["convex hull"]
        self.out_state_folder = self.sim_settings["states folder"]
        self._charge_carried = self.sim_settings["working ion"]["charge carried"]

    def start_msg(self) -> None:
        """
        Print simulation initialization message.

        This method logs the details of the simulation setup, including initial atom count,
        step probabilities, and output file configuration.
        """
        logger.info(" 0-K Voltage Profile ".center(120, "="))
        logger.info("")
        logger.info("Stating 0-K Voltage Profile simulation".center(120, " "))
        logger.info(f"Number of states = {self.n_states}")
        if self.sim_settings.get("calculation") != "vc-relax":
            logger.info(f"Optimizer: {self._optimizer_type} (fmax={self._fmax:.2e}, max steps={self._max_steps})")
        logger.info("")
        if self.save_trj_step:
            logger.info(f"Saving trajectory in `{self.out_trj}` every {self.save_trj_step} steps")
        if self.save_thermo_step:
            logger.info(f"Saving thermo in `{self.out_thermo}` every {self.save_thermo_step} steps")
        if self.save_state_step:
            logger.info(f"Saving chemical state in `{self.out_state_folder}` folder every {self.save_state_step} steps")
        logger.info("")

    def warmup(self) -> None:
        """
        Prepare the simulation.
        """
        self.initialize()
        self.start_msg()
        self._write_file_headers()
        save_dict_to_yaml(self.sim_settings, "full_simulation_settings.yaml")  # type: ignore[arg-type]

    def print_report(self):
        """
        Print the profiling report for simulation and sub-routines.
        """
        logger.info("")
        logger.info(" PERFORMANCE REPORT ".center(120, "="))
        logger.info("")
        for line in profiler_calc.report():
            logger.info(line)
        logger.info("")
        for line in profiler_io.report():
            logger.info(line)

    @profiler_calc.track
    def compute_convexhull(self):
        """
        Compute convex hull
        TODO: add more info (eq.s)
        """
        self._voltage_calculator.get_state_energy()
        self._voltage_calculator.get_n_max()
        self._voltage_calculator.get_extremes()
        self._voltage_calculator.get_reduce_factor()
        self._voltage_calculator.get_formation_energy()
        self._voltage_calculator.get_convexhull()

    @profiler_io.track
    def write_convexhull(self):
        """
        Save in a file the convex hull
        """
        self._voltage_calculator.write_convexhull(self.out_convexhull)
        logger.info(f"Save file {self.out_convexhull}")

    @profiler_calc.track
    def compute_voltage_profile(self):
        """
        Compute voltage profile from convex hull
        TODO: add more info (eq.s)
        """
        self._voltage_calculator.get_number_of_ions()
        self._voltage_calculator.get_voltage()

    @profiler_io.track
    def write_voltage(self):
        """
        Save in a file the voltage profile
        """
        self._voltage_calculator.write_voltage(self.out_voltage)
        logger.info(f"Save file {self.out_voltage}")

    def _custom_convex_hull(self):
        """
        Compute convex hull and voltage profile out of the already existing state files
        """
        files, csv_files = self.read_restart_files()
        if len(csv_files) == self.n_states:
            self.saved_state_files.extend(csv_files)
        else:
            # raise RuntimeError(
            #    f"The number of csv files found is {len(csv_files)}. However, {self.n_states} states are expected"
            # )
            logger.warning(
                f"The number of csv files found is {len(csv_files)}. However, {self.n_states} states are expected"
            )
            self.saved_state_files.extend(csv_files)

    def _custom_steps(self):
        """
        Find best atoms to remove for specific steps
        """
        if isinstance(self.sim_settings["steps_id"], int):
            self.sim_settings["steps_id"] = [self.sim_settings["steps_id"]]
        for self._i_state in self.sim_settings["steps_id"]:
            if self._i_state > self.n_states - 1:
                raise RuntimeError(
                    f"Step id: {self._i_state} is larger than the number of states {self.n_states -1}. Please choose a valid state."
                )
            if self.sim_settings["removal_method"] == "brute_force":
                num_Li_to_be_removed = self._i_state
                logger.info(
                    f"Find the lowest energy configuration when removing {num_Li_to_be_removed} Li atoms by brute force"
                )
                system = self.system.copy()
                atom_to_remove = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]
                logger.info(f" Remove {self._i_state} {self.element} atoms from {system}".center(120, "-"))
                self.delete_csv()  # delete the csv file for the step to restart
                self._remove(system, num_Li_to_be_removed, atom_to_remove)
            elif self.sim_settings["removal_method"] == "semi_brute_force":
                num_Li_to_be_removed = 1
                logger.info(
                    f"Find the lowest energy configuration when removing one Li atoms in step {self._i_state} by semi brute force"
                )
                if self._i_state == 0:
                    system = self.system.copy()
                    atom_to_remove = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]
                    num_Li_to_be_removed = 0
                else:
                    restart_file = glob.glob(f"{self.out_state_folder}/{self._i_state - 1:03d}-*.xyz")
                    self.delete_csv()  # delete the csv file for the step to restart
                    if len(restart_file) == 0:
                        logger.error(f"No restart file found for step {self._i_state}")
                        raise RuntimeError(f"No restart file found for step {self._i_state}")
                    logger.debug(f"Reading system {restart_file[0]}")
                    system = read(restart_file[0])
                    atom_to_remove = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]
                self._remove(system, num_Li_to_be_removed, atom_to_remove)
            else:
                raise RuntimeError("only brute_force and semi_brute_force methods are implemented.")

        logger.info("DONE!")

    def _custom_finish_interrupted_step(self) -> None:
        """
        Continue custome method to only finish the last interrupted step.
        """
        files, csv_files = self.read_restart_files()
        if len(csv_files) == 0:
            raise RuntimeError("No state files found to continue the simulation")

        _max_xyz_file = max(files, key=lambda x: int(x[:3]))
        _max_csv_file = max(csv_files, key=lambda x: int(x.split("/")[-1].split("-")[0]))
        self._i_state = int(_max_xyz_file[:3])
        self._continue_interrupted_step(_max_xyz_file, _max_csv_file)

    def post_process(self):
        """
        Convex hull and voltage
        """
        logger.info("Computing Convex Hull")
        self._voltage_calculator = VoltageCalculator(
            self.saved_state_files,
            self.element,
            self.chemical_potential,
            self._charge_carried,
        )
        self.compute_convexhull()
        self.write_convexhull()
        logger.info("Computing Voltage steps")
        self.compute_voltage_profile()
        self.write_voltage()
        # logger.debug(f"{self._i_state}  {self.sim_settings["plot_frequency"]}  {self._i_state%self.sim_settings["plot_frequency"]}")
        if self.sim_settings["plots"] is True and hasattr(self, "_i_state"):
            if self._i_state % self.sim_settings["plot_frequency"] == 0:
                logger.info(
                    f"Plotting convex hull and voltage profile every {self.sim_settings['plot_frequency']} steps"
                )
                self.plot_hull_and_voltage(self._i_state)

    def plot_hull_and_voltage(self, plot_name):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7))

        # =============================================
        # First subplot: Convex Hull
        # =============================================

        df_hull = pd.read_csv(self.out_convexhull)

        ax1.plot(df_hull["x"], df_hull["formation energy"], "o", color="C0", alpha=0.3)
        hull_lowest = df_hull.loc[df_hull.groupby("x")["formation energy"].idxmin()]
        ax1.plot(hull_lowest["x"], hull_lowest["formation energy"], "h-", color="red")

        ax1.set_xlabel("x")
        ax1.set_ylabel("Formation energy (eV/atom)")
        ax1.set_title("Convex Hull")
        ax1.grid(True)

        # =============================================
        # Second subplot: Voltage Profile
        # =============================================

        df_voltage = pd.read_csv(self.out_voltage)

        ax2.step(df_voltage["x1"], df_voltage["V"], "h--", where="post", color="red", linewidth=1)

        ax2.set_xlabel("x")
        ax2.set_ylabel("Voltage (V)")
        ax2.set_title("Voltage Profile")
        ax2.grid(True)

        # Adjust layout and save plot
        for ax in [ax1, ax2]:
            ax.set_xlim(-0.02, 1.02)

        plt.tight_layout()
        plt.savefig(f'{self.sim_settings["plots folder"]}/{plot_name}.pdf')

    def run(self):
        """
        Run the simulation to compute the voltage profile.

        Iteratively remove atoms and compute the energy change to determine the voltage profile.
        """
        self.warmup()

        if self.sim_settings["continue"] is False:
            self._scratch()
            self.plot_hull_and_voltage("convex_hull_voltage_profile")
        elif self.sim_settings["continue"] is True:
            # Restart from last found state
            self._restart()
            self.plot_hull_and_voltage("convex_hull_voltage_profile")
        elif self.sim_settings["continue"].upper() == "CUSTOM":
            logger.info("Continuing simulation with custom settings")
            if self.sim_settings["fully_intercalated"] is True:
                logger.info("CUSTOM: Optimizing fully intercalated configuration")
                self._i_state = 0
                self.optimize_initial_configuration()
            if self.sim_settings["fully_deintercalated"] is True:
                logger.info("CUSTOM: Optimizing fully deintercalated configuration")
                self.optimize_last_configuration()
            if self.sim_settings["finish_interrupted_step"] is True:
                logger.info("CUSTOM: Finish last interrupted step")
                self._custom_finish_interrupted_step()
            if isinstance(self.sim_settings["steps_id"], (int, list)):
                logger.info(f"CUSTOM: Continuing simulation with custom step/s {self.sim_settings['steps_id']}")
                # Find the lowest energy for a specific state
                self._custom_steps()
            if self.sim_settings["post_process"] is True:
                logger.debug("CUSTOM: Post processing")
                # Assuming you have all the state csv files, find the convex hull
                self._custom_convex_hull()
                self.post_process()
                self.plot_hull_and_voltage("convex_hull_voltage_profile")
        else:
            raise ValueError(
                f"Unknown continue option {self.sim_settings['continue']}, choose from False, True or CUSTOM"
            )

        logger.info(" END ".center(120, "="))
        self.print_report()
