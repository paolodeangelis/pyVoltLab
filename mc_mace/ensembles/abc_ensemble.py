import glob
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from ase.io import write
from loguru import logger
from numpy.random import Generator

from mc_mace.mc import MC
from mc_mace.utils.io import append_line_to_file
from mc_mace.utils.profiler import MethodProfiler

ATTEMPT_TYPE = ["volume", "position", "creation", "destruction"]

profiler_io = MethodProfiler(name="Profiling I/O")


class Ensemble(ABC):
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
        self.write_file_headers()

    def write_file_headers(self) -> None:
        if self.out_thermo is not None:
            self._out_thermo_header()
        if self.out_events is not None:
            self._out_events_header()

    def _update_step(self) -> None:
        self._i_step += 1

    def fail(self) -> None:
        self.reject[self._move_type] += 1
        self._energy_store_thermo[self._i_step % self.save_thermo_step] = self.engine.get_state_energy()  # type: ignore[operator]
        self.update_files()

    def success(self) -> None:
        self.accept[self._move_type] += 1
        self._energy_store_thermo[self._i_step % self.save_thermo_step] = self.engine.get_state_energy()  # type: ignore[operator]
        self._succ_step += 1
        self.engine.update_state()
        if self._move_type == "destruction" or self._move_type == "creation":
            self.engine.update_neighbor_list()
        self.update_event()
        self.update_files()

    def update_event(self) -> None:
        if self.out_events is not None:
            for move in ATTEMPT_TYPE:
                self.events[move].append(self.accept[move])  # type: ignore[union-attr]
            self.events["step"].append(self._i_step)  # type: ignore[union-attr]
            self.events["n_stored"] += 1  # type: ignore[operator]

    def update_files(self) -> None:
        if self.out_trj is not None and self._i_step % self.save_trj_step == 0:  # type: ignore[operator]
            self.save_trj()
        if self.out_restart is not None and self._i_step % self.save_restart_step == 0:  # type: ignore[operator]
            self.save_restart()
        if self.out_events is not None and self._i_step % self.save_events_step == 0:  # type: ignore[operator]
            self.save_events()
        if self.out_thermo is not None and self._i_step % self.save_thermo_step == 0:  # type: ignore[operator]
            self.save_thermo()
        if self.out_state_folder is not None and self._i_step % self.save_state_step == 0:  # type: ignore[operator]
            self.save_state()

    @profiler_io.track
    def save_trj(self) -> None:
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

    @profiler_io.track
    def save_state(self) -> None:
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
            elif len(files) == 1:
                file_path = files[-1]
            else:
                logger.critical(f"Found multiple state files with formula {formula}, in folder {self.out_state_folder}")
                raise ValueError(
                    f"Found multiple state files with formula {formula}, in folder {self.out_state_folder}"
                )

            append_line_to_file(
                file_path,
                f"{self._i_step:<10d}, {energy:15.8e}, {volume:15.8e}, {len(atoms):15d}",
            )
            logger.debug(f"Updated state file: {file_path}")

    def tuning(self) -> None:
        if self.tunning_step is not None:
            if self._i_step % self.tunning_step == 0 and self._i_step > 1:
                self.engine.tune_max_steps()
                self.save_restart()

    @abstractmethod
    def start_msg(self) -> None:
        logger.info("Start message")
        pass

    @abstractmethod
    def mc_step(self) -> None:
        pass

    @abstractmethod
    def mc_report(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        self.start_msg()
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
