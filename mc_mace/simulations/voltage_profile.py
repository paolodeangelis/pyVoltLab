import glob
import os
import shutil
import warnings
from io import StringIO
from itertools import combinations
from pathlib import Path

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
        if len(self._state_energy[self._formula_full]) > 1:
            print("Too many energy in full state")
            print(self._state_energy[self._formula_full])
        self._energy_full = self._state_energy[self._formula_full]
        self._n_ion_max = n_max
        if len(self._state_energy[self._formula_empty]) > 1:
            print(self._state_energy[self._formula_empty])
            print("Too many energy in empty state")
        self._energy_empty = self._state_energy[self._formula_empty]

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

    def get_voltage(self):
        self.voltage_steps = np.zeros((self._stable_points.shape[0] - 1, 3))
        e_full_per_formula = self._energy_full / self.reduce_factor
        e_empty_per_formula = self._energy_empty / self.reduce_factor
        for i in range(1, self._stable_points.shape[0]):
            x1, e1 = self._stable_points[i - 1]
            x2, e2 = self._stable_points[i]
            delta_e = e2 - e1 + (x2 - x1) * (e_full_per_formula - e_empty_per_formula - self.working_ion_energy)
            delta_x = x2 - x1
            voltage = -delta_e / (delta_x * self.charge_carried)  # Voltage in volts
            # self.voltage_steps.append([float(x1), float(x2), float(voltage)])
            self.voltage_steps[i - 1, :] = [
                x1,
                x2,
                voltage[0],
            ]  # [x1, x2, voltage] #[float(x1), float(x2), float(voltage)]

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
            prefix = "[" + f"State {self._i_state:d}/{self.n_states:d}".center(19, " ") + "] "
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

    def _restart(self) -> None:
        """
        Continue the simulation from the previous state.
        """
        if self.sim_settings["continue"]:
            file_pattern = "[0-9][0-9][0-9]-*"
            files = glob.glob(file_pattern + ".xyz", root_dir=self.out_state_folder)
            csv_files = glob.glob(file_pattern + ".csv", root_dir=self.out_state_folder)

        if len(csv_files) != len(files):
            raise RuntimeError(
                f"The number of .csv files should be the same as the number of .xyz files. Check {self.out_state_folder} or choose continue: False to start from scratch."
            )
        if not files:
            logger.info("No files found for restarting. Volta profile will start from scratch.")
            self._scratch()
        else:
            _max = max(files, key=lambda x: int(x[:3]))
            _max_csv = max(csv_files, key=lambda x: int(x[:3]))
            logger.debug(f"Save a copy of {_max_csv} to {_max_csv}_b.")
            shutil.move(f"{self.out_state_folder}/{_max_csv}", f"{self.out_state_folder}/{_max_csv}_b")
            files.remove(_max)
            _file = max(files, key=lambda x: int(x[:3]))
            logger.info(f"Continuing volta profile from {_file}")
            self.saved_state_files.extend(csv_files)
            self.state_0 = read(self.out_state_folder + "/" + _file)
            cont_state = int(_file[:3])
            for self._i_state in range(cont_state + 1, self.n_states):
                _energy = self._find_atom_to_remove()

                self.update_files(_energy)
                self.update_states()

    def _scratch(self) -> None:
        """
        Start the simulation from scratch.
        """
        for self._i_state in range(self.n_states):
            logger.info(f" State {self._i_state} ".center(120, "-"))
            if self._i_state == 0:
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
                force_max_start = np.max(np.linalg.norm(self.state_1.get_forces(), axis=1))
                cell_start = self.state_1.cell.cellpar()
                vol_start = self.state_1.cell.volume
                logger.debug(f"optimizing system {self.state_1.get_chemical_formula()}")
                self.state_1, energy_end = self._optimize_system(self.state_1)
                force_max_end = np.max(np.linalg.norm(self.state_1.get_forces(), axis=1))
                cell_end = self.state_1.cell.cellpar()
                vol_end = self.state_1.cell.volume
                logger.info(self.__logger_prefix() + f"Energy {energy_start:.3f} eV -> {energy_end:.3f} eV")
                logger.info(
                    self.__logger_prefix() + f"Max Force {force_max_start:.3e} eV/A -> {force_max_end:.3e} eV/A"
                )
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
            else:
                energy_end = self._find_atom_to_remove()

            self.update_files(energy_end)
            self.update_states()

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
                directory=self.sim_settings["QE_dir"] + f"/{sub_file}",
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
                logger.debug(f"mu({element}) = {mu:.3f} eV")
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
            f"{'step'},{'energy'},{'volume'},{'atoms'}",
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
                logger.warning(
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
            num_Li_to_be_removed = self.n_states - len(
                np.where(self.state_0.get_atomic_numbers() == atomic_numbers[self.element])[0]
            )
            return self._remove(system, num_Li_to_be_removed)
        elif self.sim_settings["removal_method"] == "semi_brute_force":
            logger.info("Semi brute force method")
            system = self.state_0.copy()
            num_Li_to_be_removed = 1
            return self._remove(system, num_Li_to_be_removed)
        elif self.sim_settings["removal_method"] == "genetic":
            logger.info("Genetic algorithm method")
            raise NotImplementedError("Genetic algorithm not implemented")
        elif self.sim_settings["removal_method"] == "cluster_expansion":
            logger.info("Cluster expansion method not implemented yet")
            raise NotImplementedError("Cluster expansion not implemented")

    def _remove(self, system: Atoms, num_Li_to_be_removed: int):
        """
        Identify the atom to remove by computing energy differences.

        Returns:
            int: Index of the atom to remove.
            float: The minimum total energy obtained after removing one atom.
        """
        min_energy = float("inf")
        best = None
        best_ai = None
        atom_to_remove = np.where(system.get_atomic_numbers() == atomic_numbers[self.element])[0]
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
                tag += str(val)
            self._set_calculator(tag + "_" + self.state_1.get_chemical_formula())
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
        self._voltage_calculator.get_voltage()

    @profiler_io.track
    def write_voltage(self):
        """
        Save in a file the voltage profile
        """
        self._voltage_calculator.write_voltage(self.out_voltage)
        logger.info(f"Save file {self.out_voltage}")

    def run(self):
        """
        Run the simulation to compute the voltage profile.

        Iteratively remove atoms and compute the energy change to determine the voltage profile.
        """
        self.warmup()

        if self.sim_settings["continue"]:
            self._restart()
        else:
            self._scratch()

        logger.info("Computing Convex Hull")
        self._voltage_calculator = VoltageCalculator(
            self.saved_state_files, self.element, self.chemical_potential, self._charge_carried
        )
        self.compute_convexhull()
        self.write_convexhull()
        logger.info("Computing Voltage steps")
        self.compute_voltage_profile()
        self.write_voltage()
        logger.info(" END ".center(120, "="))
        self.print_report()
