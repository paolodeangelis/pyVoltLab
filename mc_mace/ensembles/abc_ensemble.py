import glob
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.io import write
from loguru import logger
from numpy.random import Generator

from mc_mace.mc import MC
from mc_mace.utils.io import append_line_to_file
from mc_mace.utils.profiler import MethodProfiler

ATTEMPT_TYPE = ["volume", "position", "creation", "destruction"]

profiler_io = MethodProfiler(name="Profiling I/O")


class Ensemble(ABC):
    """
    Abstract base class for Monte Carlo simulations with different ensembles.

    Attributes:
        engine (MC): The Monte Carlo engine handling the simulation state.
        steps (int): The total number of steps in the simulation.
        random_number_gen (Generator | None): A random number generator instance (default: numpy default RNG).
        out_thermo (str | Path | None): Path to the output file for thermodynamic properties.
        out_trj (str | Path | None): Path to the output trajectory file.
        out_events (str | Path | None): Path to the output events file.
        out_state_folder (str | Path | None): Path to the folder for saving state files.
        out_restart (str | Path | None): Path to the output restart file.
        save_trj_step (int | None): Frequency (in steps) to save trajectory data.
        save_thermo_step (int | None): Frequency (in steps) to save thermodynamic data.
        save_events_step (int | None): Frequency (in steps) to save events data.
        save_state_step (int | None): Frequency (in steps) to save state data.
        save_restart_step (int | None): Frequency (in steps) to save restart files.
        tunning_step (int | None): Frequency (in steps) to tune simulation parameters.
    """

    def __init__(
        self,
        engine: MC,
        steps: int,
        random_number_gen: Generator | None = None,
        out_thermo: str | Path | None = None,
        out_trj: str | Path | None = None,
        out_events: str | Path | None = None,
        out_state_folder: str | Path | None = None,
        out_restart: str | Path | None = None,
        save_trj_step: int | None = None,
        save_thermo_step: int | None = None,
        save_events_step: int | None = None,
        save_state_step: int | None = None,
        save_restart_step: int | None = None,
        tunning_step: int | None = None,
    ) -> None:
        """
        Initialize the `Ensemble` class.

        Args:
            engine (MC): The Monte Carlo engine handling simulation state and moves.
            steps (int): Total number of steps for the simulation.
            random_number_gen (Generator | None): Random number generator (default: numpy default RNG).
            out_thermo (str | Path | None): File path for thermodynamic property output.
            out_trj (str | Path | None): File path for trajectory output.
            out_events (str | Path | None): File path for events output.
            out_state_folder (str | Path | None): Folder path for state file storage.
            out_restart (str | Path | None): File path for restart file output.
            save_trj_step (int | None): Frequency of saving trajectory data (steps).
            save_thermo_step (int | None): Frequency of saving thermodynamic data (steps).
            save_events_step (int | None): Frequency of saving events data (steps).
            save_state_step (int | None): Frequency of saving state data (steps).
            save_restart_step (int | None): Frequency of saving restart files (steps).
            tunning_step (int | None): Frequency of tuning simulation parameters (steps).
        """
        self.steps = steps
        self._i_step: int = 0
        self._i_start: int = 0
        self.out_thermo = out_thermo
        self.out_trj = out_trj
        self.out_events = out_events
        self.out_state_folder = out_state_folder
        self.out_restart = out_restart
        self.save_trj_step = save_trj_step
        self.save_thermo_step = save_thermo_step
        self.save_events_step = save_events_step
        self.save_state_step = save_state_step
        self.save_restart_step = save_restart_step
        self.tunning_step = tunning_step
        if random_number_gen is None:
            logger.warning("No random number generator set, will be used the default `numpy` generator")
            self.rng = np.random.default_rng()
        else:
            self.rng = random_number_gen
        self._succ_step: int = 0
        self.accept: dict[str, int] = {t: 0 for t in ATTEMPT_TYPE + ["nothing"]}
        self.reject: dict[str, int] = {t: 0 for t in ATTEMPT_TYPE + ["nothing"]}
        if out_events is not None:
            self.events: dict[str, int | list[int]] = {t: [] for t in ATTEMPT_TYPE + ["nothing"]}
            self.events["n_stored"] = 0
            self.events["step"] = []
        self.engine = engine
        self._move_type: str = ""
        if self.save_thermo_step is not None:
            self._energy_store_thermo = np.ones(shape=(self.save_thermo_step,)) * np.nan
        self._sim_stat: dict[str, float] = {
            "n": 0.0,
            "sumE": 0.0,
            "sumE2": 0.0,
            "sumV": 0.0,
            "sumV2": 0.0,
            "sumN": 0.0,
            "sumN2": 0.0,
        }
        for el in self.engine.insert_elements:
            self._sim_stat[f"sumN({el})"] = 0.0
            self._sim_stat[f"sumN2({el})"] = 0.0
        self.write_file_headers()
        self._new_state: bool = False
        self.allowed_steps: list["str"] = []

    # def _check_values(self) -> None:
    #     saving_steps = [
    #     self.save_trj_step,
    #     self.save_thermo_step,
    #     self.save_events_step,
    #     self.save_state_step,
    #     self.save_restart_step,
    #     ]
    #     for v_ in saving_steps:
    #         if v_ is None:
    #             v_ = 1
    #         elif not isinstance(v_, int):
    #             v_ = 1
    #         elif v_ < 1:
    #             v_ = 1

    def write_file_headers(self) -> None:
        """
        Write headers to output files for thermo and events data.

        This method ensures that all required output files start with the appropriate
        header information, enabling structured data logging during the simulation.
        """
        if self.out_thermo is not None:
            self._out_thermo_header()
        if self.out_events is not None:
            self._out_events_header()

    def _update_step(self) -> None:
        self._i_step += 1

    def _update_stats(self) -> None:
        En = self.engine.get_state_energy()
        V = self.engine.get_state_volume()
        conf_atoms = self.engine.get_state_configuration()
        N = len(conf_atoms)
        Ne: dict[str, float] = {}
        for el in self.engine.insert_elements:
            Ne[el] = len(np.where(conf_atoms.get_atomic_numbers() == atomic_numbers[el])[0])
        self._sim_stat["sumE"] += En
        self._sim_stat["sumE2"] += En**2
        self._sim_stat["sumV"] += V
        self._sim_stat["sumV2"] += V**2
        self._sim_stat["sumN"] += float(N)
        self._sim_stat["sumN2"] += float(N) ** 2
        for el in self.engine.insert_elements:
            self._sim_stat[f"sumN({el})"] += float(Ne[el])
            self._sim_stat[f"sumN2({el})"] += float(Ne[el] ** 2)
        self._sim_stat["n"] += 1.0

    def get_simulation_stat(self, property: str) -> tuple[float, float]:
        """
        Get mean and variance of a simulation property.

        Args:
            property (str): The property to retrieve statistics for (e.g., "energy", "volume", "atoms", or element symbol).

        Returns:
            tuple[float, float]: Mean and variance of the specified property.
        """
        n = self._sim_stat["n"]
        if property.lower() == "energy":
            mean = self._sim_stat["sumE"] / n
            mean2 = self._sim_stat["sumE2"] / n
        elif property.lower() == "volume":
            mean = self._sim_stat["sumV"] / n
            mean2 = self._sim_stat["sumV2"] / n
        elif property.lower() == "atoms":
            mean = self._sim_stat["sumN"] / n
            mean2 = self._sim_stat["sumN2"] / n
        else:
            # assume it is an element
            el = property
            mean = self._sim_stat[f"sumN({el})"] / n
            mean2 = self._sim_stat[f"sumN2({el})"] / n
        return mean, mean2 - mean**2

    def fail(self) -> None:
        """
        Handle a failed Monte Carlo move.

        This updates rejection statistics, stores energy information, updates simulation statistics,
        and updates output files.
        """
        self.reject[self._move_type] += 1
        self._energy_store_thermo[self._i_step % self.save_thermo_step] = self.engine.get_state_energy()  # type: ignore[operator]
        self._update_stats()
        self.update_files()

    def success(self) -> None:
        """
        Handle a successful Monte Carlo move.

        This updates acceptance statistics, stores energy information, updates the engine state,
        manages neighbor lists, and updates output files.
        """
        self.accept[self._move_type] += 1
        self._energy_store_thermo[self._i_step % self.save_thermo_step] = self.engine.get_state_energy()  # type: ignore[operator]
        self._succ_step += 1
        if len(self.engine.atoms_new) != len(self.engine.atoms_old):
            self._new_state = True
        self.engine.update_state()
        if self._move_type == "destruction" or self._move_type == "creation":
            self.engine.update_neighbor_list()
        self.update_event()
        self._update_stats()
        self.update_files()
        self._new_state = False  # Reset flag

    def update_event(self) -> None:
        """
        Update events-related statistics and store them in the events output.
        """
        if self.out_events is not None:
            for move in ATTEMPT_TYPE:
                self.events[move].append(self.accept[move])  # type: ignore[union-attr]
            self.events["step"].append(self._i_step)  # type: ignore[union-attr]
            self.events["n_stored"] += 1  # type: ignore[operator]

    def update_files(self) -> None:
        """
        Update all output files according to the current simulation step.

        This method manages writing data to trajectory, restart, thermo, state, and events files
        based on their respective save frequencies.
        """
        if self.out_trj is not None and self._i_step % self.save_trj_step == 0:  # type: ignore[operator]
            self.save_trj()
        if self.out_restart is not None and self._i_step % self.save_restart_step == 0:  # type: ignore[operator]
            self.save_restart()
        if self.out_events is not None and self._i_step % self.save_events_step == 0:  # type: ignore[operator]
            self.save_events()
        if self.out_thermo is not None and self._i_step % self.save_thermo_step == 0:  # type: ignore[operator]
            self.save_thermo()
        if self.out_state_folder is not None and self._i_step % self.save_state_step == 0 or self._new_state:  # type: ignore[operator]
            self.save_state()

    @profiler_io.track
    def save_trj(self) -> None:
        """
        Save the current state of the simulation to the trajectory file.

        This method writes atomic configuration, energy, volume, and step information
        to the trajectory file at the specified save frequency.

        Raises:
            ValueError: If the trajectory file path is not set.
        """
        atoms = self.engine.get_state_configuration()
        energy = self.engine.get_state_energy()
        volume = self.engine.get_state_volume()
        atoms.info["volume"] = volume
        atoms.info["potential_energy"] = energy
        atoms.info["energy"] = energy
        atoms.info["step"] = self._i_step
        atoms.info["success_step"] = self._succ_step
        write(self.out_trj, atoms, append=True)
        logger.debug(f"Updated trajectory file: {self.out_trj}")

    @profiler_io.track
    def save_restart(self) -> None:
        """
        Save the current simulation state to the restart file.

        The restart file contains atomic configuration, energy, volume, and simulation
        metadata, allowing the simulation to be resumed later.

        Raises:
            ValueError: If the restart file path is not set.
        """
        atoms = self.engine.get_state_configuration()
        energy = self.engine.get_state_energy()
        volume = self.engine.get_state_volume()
        atoms.info["volume"] = volume
        atoms.info["potential_energy"] = energy
        atoms.info["energy"] = energy
        atoms.info["step"] = self._i_step
        atoms.info["success_step"] = self._succ_step
        atoms.info["max_displacement"] = self.engine.max_step["position"]
        atoms.info["max_volume_change"] = self.engine.max_step["volume"]
        write(self.out_restart, atoms, append=False)
        logger.info(f"Saving restart file {self.out_restart}")

    def _out_events_header(self) -> None:
        if self.out_events is not None:
            append_line_to_file(self.out_events, ",".join(f"{t}" for t in ["step"] + ATTEMPT_TYPE))

    @profiler_io.track
    def save_events(self) -> None:
        """
        Save simulation events to the events file.

        The events file logs the number of accepted moves for each move type at
        specific simulation steps. It helps track move statistics during the simulation.

        Raises:
            ValueError: If the events file path is not set.
        """
        if self.out_events is not None:
            lines = []
            for i in range(self.events["n_stored"]):  # type: ignore[arg-type]
                lines.append(",".join(f"{self.events[t][i]:10d}" for t in ["step"] + ATTEMPT_TYPE))  # type: ignore[index]
            append_line_to_file(self.out_events, lines)
            self.events = {t: [] for t in ATTEMPT_TYPE}
            self.events["n_stored"] = 0
            self.events["step"] = []
            logger.debug(f"Updated events file: {self.out_events}")

    def _out_thermo_header(self) -> None:
        if self.out_thermo is not None:
            append_line_to_file(
                self.out_thermo,
                f"{'step'},{'energy'},{'mean energy'},{'std energy'},{'volume'},{'atoms'}",
            )

    @profiler_io.track
    def save_thermo(self) -> None:
        """
        Save thermodynamic properties to the thermo file.

        This method writes energy, volume, and atom count statistics to the thermo file
        at specified save frequencies.

        Raises:
            ValueError: If the thermo file path is not set.
        """
        if self.out_thermo is not None:
            atoms = self.engine.get_state_configuration()
            energy = self.engine.get_state_energy()
            volume = self.engine.get_state_volume()
            mean_energy = np.nanmean(self._energy_store_thermo)
            if self.save_thermo_step > 1 and self._i_step - self._i_start > 0:  # type: ignore[operator]
                std_energy = np.nanstd(self._energy_store_thermo)
            else:
                std_energy = 0.0
            append_line_to_file(
                self.out_thermo,
                (
                    f"{self._i_step:10d}, {energy:15.8e}, {mean_energy:15.8e}, "
                    f"{std_energy:15.8e}, {volume:15.8e}, {len(atoms):15d}"
                ),
            )
            logger.debug(f"Updated thermo file: {self.out_thermo}")

    def _state_file_header(self, file_path: str | Path) -> None:
        append_line_to_file(
            file_path,
            f"{'step'},{'energy'},{'volume'},{'atoms'}",
        )

    def _save_state_xyz(self, file_path: str) -> None:
        atoms = self.engine.get_state_configuration()
        energy = self.engine.get_state_energy()
        volume = self.engine.get_state_volume()
        atoms.info["volume"] = volume
        atoms.info["potential_energy"] = energy
        atoms.info["energy"] = energy
        atoms.info["step"] = self._i_step
        atoms.info["success_step"] = self._succ_step
        atoms.info["max_displacement"] = self.engine.max_step["position"]
        atoms.info["max_volume_change"] = self.engine.max_step["volume"]
        write(file_path, atoms, append=False)
        logger.info(f"Saving state file {file_path}")

    @profiler_io.track
    def save_state(self) -> None:
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
            atoms = self.engine.get_state_configuration()
            energy = self.engine.get_state_energy()
            volume = self.engine.get_state_volume()
            formula = atoms.get_chemical_formula(mode="hill", empirical=False)
            files = glob.glob(os.path.join(str(self.out_state_folder), f"*{formula}.csv"))
            if len(files) < 1:
                id = len(glob.glob(os.path.join(str(self.out_state_folder), "*.csv")))
                file_path = os.path.join(str(self.out_state_folder), f"{id:03d}-{formula}.csv")
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

            append_line_to_file(
                file_path,
                f"{self._i_step:<10d}, {energy:15.8e}, {volume:15.8e}, {len(atoms):15d}",
            )
            self._save_state_xyz(conf_file_path)
            logger.debug(f"Updated state file: {file_path}")

    def tuning(self) -> None:
        if self.tunning_step is not None:
            if self._i_step % self.tunning_step == 0 and self._i_step > 1:
                self.engine.tune_max_steps()
                self.save_restart()

    def print_step_frequencies(self) -> None:
        frequencies_str = ""
        for attempt in self.allowed_steps:
            tot = self.engine.accept[attempt] + self.engine.reject[attempt]
            freq_a = self.engine.accept[attempt] / (self._i_step - self._i_start + 1)
            freq_t = tot / (self._i_step - self._i_start + 1)
            frequencies_str += f"{attempt} a: {freq_a*100:.3g}% (t:{freq_t*100:.3g}%) "
        logger.debug(frequencies_str.center(120, " "))

    @abstractmethod
    def start_msg(self) -> None:
        """
        Print or log a starting message for the simulation.

        Subclasses must implement this method to provide simulation-specific startup information.
        """
        logger.info("Start message")
        pass

    @abstractmethod
    def mc_step(self) -> None:
        """
        Perform a single Monte Carlo step.

        Subclasses must implement this method to define the logic for a single MC step,
        including state updates, move proposals, and acceptance checks.

        Returns:
            bool: True if the move is accepted, False otherwise.
        """
        pass

    @abstractmethod
    def mc_report(self) -> None:
        """
        Generate and print a detailed report for the Monte Carlo simulation.

        Subclasses must implement this method to summarize simulation results, including
        performance metrics, acceptance rates, and other relevant statistics.
        """
        pass

    def print_report(self) -> None:
        """
        Print the profiling report for MC steps and sub-routines.
        """
        logger.info("")
        logger.info(" PERFORMANCE REPORT ".center(120, "="))
        logger.info("")
        self.mc_report()
        logger.info("")
        self.engine.print_report()
        logger.info("")
        for line in profiler_io.report():
            logger.info(line)

    def run(self) -> None:
        """
        Execute the simulation over the specified number of steps.

        This method performs the Monte Carlo simulation loop, managing state updates,
        file outputs, tuning, and periodic reporting.
        """
        self.start_msg()
        for self._i_step in range(self._i_start, self.steps + 1):
            accepted = self.mc_step()

            if accepted:
                self.success()
            else:
                self.fail()
            self.print_step_frequencies()

            if self.tunning_step is not None:
                self.tuning()
        logger.info(" END ".center(120, "="))
        self.print_report()
