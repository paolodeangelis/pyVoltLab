from pathlib import Path
from typing import Any

import numpy as np
from ase.data import atomic_numbers
from ase.units import kB
from loguru import logger

from mc_mace.ensembles import muPT
from mc_mace.utils.io import (
    append_line_to_file,
    create_file_with_backup,
    save_dict_to_yaml,
)
from mc_mace.utils.moving_avg import ForgetfulMovingAvg
from mc_mace.utils.parse_input import ignore_mc_input, overwrite_mc_input
from mc_mace.utils.parse_pid_input import parse_yaml_pid_input

from .simple_mc import Simulation
from .simulation_abc import get_chem_pot


class PIDTuning(Simulation):
    """
    Monte Carlo Simulation with PID-Controlled Chemical Potential Tuning.

    Attributes:
        pid_file (Path | str): Path to the PID tuning settings file.
        pid_settings (dict | None): Parsed PID tuning settings.
        out_pid (str | None): Path to the PID output file.
        save_out_step (int): Frequency for saving PID output.
        element (str): The chemical element being tuned.
        alpha (float): Compressibility scaling factor for PID control.
        n_target (int | None): Target number of atoms for the tuned element.
        mc_steps (int): Number of Monte Carlo steps per PID update.
        beta (float): Inverse temperature factor (1 / kB * T).
        mu_t (ForgetfulMovingAvg): Moving average for chemical potentials.
        k_t (ForgetfulMovingAvg): Moving average for compressibility.
        n_t (ForgetfulMovingAvg): Moving average for atom counts.
    """

    def __init__(
        self,
        input_file: Path | str,
        pid_file: Path | str,
        log_file: Path | str,
        log_level: str,
        colorize: bool,
        device: str,
    ):
        """
        Initialize the PIDTuning simulation.

        Args:
            input_file (Path | str): Path to the YAML input file.
            pid_file (Path | str): Path to the PID tuning settings file.
            log_file (Path | str): Path to the log file.
            log_level (str): Logging verbosity level (e.g., "DEBUG", "INFO").
            colorize (bool): Whether to enable colored logging.
            device (str): Computational device to use (e.g., "cuda" or "cpu").
        """
        super().__init__(input_file, log_file, log_level, colorize, device)
        self.pid_file = pid_file
        self.pid_settings: dict[Any, Any] | None = None
        self.out_pid: str | Path | None = None
        self.save_out_step: int = 0
        self.element: str = ""
        self.alpha: float = 0.0
        self.n_target: int | str | None = None
        self.mc_steps: int = 1
        self.beta: float = 0.0
        self._t: int = 0
        self.mu_t: ForgetfulMovingAvg | None = None
        self.k_t: ForgetfulMovingAvg | None = None
        self.n_t: ForgetfulMovingAvg | None = None
        self.k_fluc: float | None = None
        self.k_min: float | None = None
        self.k_max: float | None = None
        self.early_stop_start: int = 1000
        self.early_stop_dropout: int
        self.early_stop_n: ForgetfulMovingAvg | None = None
        self.early_stop_mu: ForgetfulMovingAvg | None = None

    def initialize(self) -> None:
        """
        Initialize the PID tuning simulation.

        This includes parsing the PID settings, creating output files, and initializing the simulation.
        """
        super().initialize()
        self.pid_settings = parse_yaml_pid_input(self.pid_file)
        self.out_pid = self.pid_settings["output file"]
        self.save_out_step = self.pid_settings["saving step"]
        if self.out_pid is not None:
            create_file_with_backup(self.out_pid)
            self.__out_pid_header()
        self.n_target = self.pid_settings["n target"]
        self.mc_steps = self.pid_settings["MC steps"]
        _ = self._compute_chemical_potentials()  # run early because we need the element
        if self.n_target == "from input":
            self.n_target = len(np.where(self.system.get_atomic_numbers() == atomic_numbers[self.element])[0])  # type: ignore[union-attr]
            logger.debug(f"Target number of {self.element} from initial configuration = {self.n_target}")
        if self.pid_settings["bonds"] is not None:
            if self.system is not None:
                bonds = self.pid_settings["bonds"]
                n_max = (
                    self.n_target
                    + bonds
                    + len(np.where(self.system.get_atomic_numbers() != atomic_numbers[self.element])[0])
                )
                n_min = (
                    self.n_target
                    - bonds
                    + len(np.where(self.system.get_atomic_numbers() != atomic_numbers[self.element])[0])
                )
                self.sim_settings = overwrite_mc_input(self.sim_settings, "max atoms", n_max)  # type: ignore[arg-type]
                self.sim_settings = overwrite_mc_input(self.sim_settings, "min atoms", n_min)
            else:
                logger.error("No atomistic object load...")
                raise RuntimeError("No atomistic object load...")

    def _compute_chemical_potentials(self) -> tuple[list[str], list[float]]:
        """
        Compute chemical potentials based on PID settings.

        Returns:
            tuple[list[str], list[float]]: A tuple containing a list of elements and their chemical potentials.
        """
        if self.sim_settings is None or self.pid_settings is None:
            logger.warning("Wrong class execution, run the `initialize` method first")
            self.initialize()
        elements = []
        potentials = []
        ignore_mc_input(self.sim_settings, "chemical potential", "override by PID settings", None)  # type: ignore[arg-type]
        if len(list(self.pid_settings["chemical potential"].keys())) > 1:  # type: ignore[index]
            logger.error("Only one chemical potential at the time can be tuned")
            logger.error(f"Found more that one chemical potential in {self.pid_file}")
            logger.error(f"chemical potential: {self.pid_settings['chemical potential']}")  # type: ignore[index]
            raise NotImplementedError("Only one chemical potential at the time can be tuned")
        for element, value in self.pid_settings["chemical potential"].items():  # type: ignore[index]
            elements.append(element)
            if isinstance(value, str):
                mu = get_chem_pot(value, self.calculator)
                self.sim_settings["chemical potential"][element] = mu  # type: ignore[index]
                potentials.append(mu)
            else:
                potentials.append(value)
        self.element = elements[0]
        return elements, potentials

    def warmup(self) -> None:
        """
        Prepare the simulation and configure the ensemble for PID tuning.
        """
        self.initialize()
        self._set_engine()
        ensemble_settings = self._get_ensemble_settings()
        if self.sim_settings["ensemble"].lower() == "npt":  # type: ignore[index]
            logger.error("Only μVT or μPT ensemble is allowed in PID chemical potential tuning.")
            raise ValueError("Only μVT or μPT ensemble is allowed in PID chemical potential tuning.")
        elif self.sim_settings["ensemble"].lower() == "mupt":  # type: ignore[index]
            self.ensemble = muPT(**ensemble_settings)  # type: ignore[assignment]
        else:
            raise ValueError("Unsupported ensemble type")
        save_dict_to_yaml(self.sim_settings, "full_simulation_settings.yaml")  # type: ignore[arg-type]
        save_dict_to_yaml(self.pid_settings, "full_pid_settings.yaml")  # type: ignore[arg-type]

    def _early_stop_setup(self) -> None:
        if "early stop" in self.pid_settings:  # type: ignore[operator]
            early_stop_dict = self.pid_settings["early stop"]  # type: ignore[index]
            self.early_stop_dropout = early_stop_dict["dropout"]
            if early_stop_dict["target atoms mean"] is not False:
                logger.debug("PID early stop `target atoms mean`: ON")
                self.early_stop_n = ForgetfulMovingAvg(dropout=self.early_stop_dropout)
            if early_stop_dict["target atoms variance"] is not False:
                logger.debug("PID early stop `target atoms variance`: ON")
            if early_stop_dict["chemical potential variance"] is not False:
                logger.debug("PID early stop `chemical potential variance`: ON")
                self.early_stop_mu = ForgetfulMovingAvg(dropout=self.early_stop_dropout)
        else:
            logger.warning("PID early stop not used!")

    def pid_initialize(self) -> None:
        """
        Initialize PID-specific parameters such as compressibility, moving averages, and target values.
        """
        self.alpha = self.pid_settings["compressibility scale"]  # type: ignore[index]
        dropout = self.pid_settings["dropout"]  # type: ignore[index]
        T = float(self.sim_settings["temperature"])  # type: ignore[index]
        self.beta = 1 / (kB * T)
        self._t = 0
        self.mu_t = ForgetfulMovingAvg(dropout)
        self.k_t = ForgetfulMovingAvg(dropout)
        self.n_t = ForgetfulMovingAvg(dropout)
        self._early_stop_setup()

    def __out_pid_header(self) -> None:
        if self.out_pid is not None:
            header = [
                "step",
                "pid step",
                "mu",
                "mu_mean",
                "mu_var",
                "N",
                "N_mean",
                "N_var",
                "k",
                "k_mean",
                "k_var",
                "k_fluc",
                "k_min",
                "k_max",
            ]
            append_line_to_file(self.out_pid, ",".join(f"{t}" for t in header))

    def update_file(self) -> None:
        """
        Update the PID output file at the specified save frequency.
        """
        if self.out_pid is not None and self._t % self.save_out_step == 0:
            self.save_out()

    def save_out(self) -> None:
        """
        Save the current state of the PID tuning to the output file.
        """
        if self.out_pid is not None:
            append_line_to_file(
                self.out_pid,
                (
                    f"{self.ensemble._i_step:10d}, {self._t:10d}, "  # type: ignore[attr-defined]
                    f"{self.mu_t.get_last():15.8e}, {self.mu_t.get_mean():15.8e}, {self.mu_t.get_variance():15.8e}, "  # type: ignore[union-attr]
                    f"{self.n_t.get_last():15.8e}, {self.n_t.get_mean():15.8e}, {self.n_t.get_variance():15.8e}, "  # type: ignore[union-attr]
                    f"{self.k_t.get_last():15.8e}, {self.k_t.get_mean():15.8e}, {self.k_t.get_variance():15.8e},"  # type: ignore[union-attr]
                    f"{self.k_fluc:15.8e}, {self.k_min:15.8e}, {self.k_max:15.8e}"
                ),
            )
            logger.debug(f"Updated PID out file: {self.out_pid}")

    def __logger_prefix(self) -> str:
        return "[" + f"PID Step {self._t:d}".center(19, " ") + "] "

    def check_early_stop(self) -> bool:
        early_stop = False
        checks: list[bool] = []
        if "early stop" in self.pid_settings and self._t > self.early_stop_start:  # type: ignore[operator]
            early_stop_dict = self.pid_settings["early stop"]  # type: ignore[index]
            logger.debug(self.__logger_prefix() + " PID Early Stop check ".center(120, " "))
            if early_stop_dict["target atoms mean"] is not False:
                checks.append(
                    abs(self.early_stop_n.get_mean() - self.n_target) < float(early_stop_dict["target atoms mean"]) and (abs(self.early_stop_n.get_mean() - self.n_target) > 1e-20)  # type: ignore[union-attr, operator]
                )
                logger.debug(
                    self.__logger_prefix()
                    + (
                        f" * target atoms mean = {checks[-1]} "
                        f"(<N> - N_0 = {abs(self.early_stop_n.get_mean() - self.n_target):.3f}) "  # type: ignore[union-attr, operator]
                        f"< {float(early_stop_dict['target atoms mean']):1.3e})"
                    )
                    .ljust(60, " ")
                    .center(120, " ")
                )
            if early_stop_dict["target atoms variance"] is not False:
                checks.append(
                    abs(self.early_stop_n.get_variance()) < float(early_stop_dict["target atoms variance"]) and abs(self.early_stop_n.get_variance()) > 1e-20  # type: ignore[union-attr]
                )
                logger.debug(
                    self.__logger_prefix()
                    + (
                        f" * target atoms variance = {checks[-1]} "
                        f"(<N^2> = {self.early_stop_n.get_variance():.3e} "  # type: ignore[union-attr]
                        f"< {float(early_stop_dict['target atoms variance']):1.3e})"
                    )
                    .ljust(60, " ")
                    .center(120, " ")
                )
            if early_stop_dict["chemical potential variance"] is not False:
                checks.append(
                    abs(self.early_stop_mu.get_variance()) < float(early_stop_dict["chemical potential variance"]) and abs(self.early_stop_mu.get_variance()) > 1e-20  # type: ignore[union-attr]
                )
                logger.debug(
                    self.__logger_prefix()
                    + (
                        f" * chemical potential variance = {checks[-1]} "
                        f"(<μ^2> = {self.early_stop_mu.get_variance():.3e} "  # type: ignore[union-attr]
                        f"< {float(early_stop_dict['chemical potential variance']):1.3e})"
                    )
                    .ljust(60, " ")
                    .center(120, " ")
                )
            early_stop = np.all(checks)
        return early_stop

    def run(self) -> None:
        """
        Execute the Monte Carlo simulation with PID-controlled chemical potential tuning.
        """
        self.warmup()
        self.pid_initialize()
        while self.ensemble._i_step < self.ensemble.steps:  # type: ignore[attr-defined]
            logger.info(f" PID step {self._t:d} ".center(15, " ").center(120, "+"))
            for i in range(self.mc_steps):
                accepted = self.ensemble.mc_step()  # type: ignore[attr-defined]

                if accepted:
                    self.ensemble.success()  # type: ignore[attr-defined]
                else:
                    self.ensemble.fail()  # type: ignore[attr-defined]
                self.ensemble.print_step_frequencies()  # type: ignore[attr-defined]

                if self.ensemble.tunning_step is not None:  # type: ignore[attr-defined]
                    self.ensemble.tuning()  # type: ignore[attr-defined]
                self.ensemble._i_step += 1  # type: ignore[attr-defined]
            atoms = self.ensemble.engine.get_state_configuration()  # type: ignore[attr-defined]
            self.n_t.add_sample(len(np.where(atoms.get_atomic_numbers() == atomic_numbers[self.element])[0]))  # type: ignore[union-attr]
            self.mu_t.add_sample(self.ensemble.engine.mus[0])  # type: ignore[attr-defined,union-attr]
            if self.early_stop_n is not None:
                self.early_stop_n.add_sample(
                    len(np.where(atoms.get_atomic_numbers() == atomic_numbers[self.element])[0])
                )
            if self.early_stop_mu is not None:
                self.early_stop_mu.add_sample(self.ensemble.engine.mus[0])  # type: ignore[attr-defined]
            if self.check_early_stop():
                logger.info("Simulation early stopped")
                break
            n_mean = self.n_t.get_mean()  # type: ignore[union-attr]
            mu_mean = self.mu_t.get_mean()  # type: ignore[union-attr]
            n_var = self.n_t.get_variance()  # type: ignore[union-attr]
            mu_var = self.mu_t.get_variance()  # type: ignore[union-attr]

            self.k_fluc = self.beta * n_var
            self.k_min = self.alpha / np.sqrt(self._t + 1)
            if np.abs(mu_var) > 0.0:
                self.k_max = np.sqrt(n_var / mu_var)
            else:
                self.k_max = 0.0
            self.k_t.add_sample(max(self.k_min, min(self.k_max, self.k_fluc)))  # type: ignore[union-attr,type-var,arg-type]
            k_mean = self.k_t.get_mean()  # type: ignore[union-attr]
            k_var = self.k_t.get_variance()  # type: ignore[union-attr]
            logger.debug(
                self.__logger_prefix()
                + f"compressibility k_t = {self.k_t.get_last():.3g} 1/eV (k_fluc={self.k_fluc:.3g} 1/eV, k_min = {self.k_min:.3g} 1/eV, k_max = {self.k_max:.3g} 1/eV)"  # type: ignore[union-attr]
            )
            mu_new = self.mu_t.get_mean() + (self.n_target - n_mean) / self.k_t.get_last()  # type: ignore[union-attr,operator]
            self.ensemble.engine.mus = [mu_new]  # type: ignore[attr-defined]
            logger.info(
                self.__logger_prefix()
                + f"updating chemical potential μ = {self.mu_t.get_last():.3f} eV -> μ  ={mu_new:.3f} eV"  # type: ignore[union-attr]
            )
            logger.debug(
                self.__logger_prefix()
                + f"<μ> ={mu_mean:.3g} eV, <μ2> = {mu_var:.3g} eV, <k> ={k_mean:.3g}, <k2> = {k_var:.3g}, <N> = {n_mean:.3g}, <N2> = {n_var:.3g} (N_target = {self.n_target:d})"
            )
            self.update_file()
            self._t += 1

        logger.info(" END ".center(120, "="))
        self.ensemble.print_report()  # type: ignore[attr-defined]
