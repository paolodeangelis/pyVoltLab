import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
from loguru import logger

from mc_mace.mc import MC
from mc_mace.utils.header import print_header
from mc_mace.utils.io import (
    clean_ase_read,
    create_file_with_backup,
    create_folder_with_backup,
)
from mc_mace.utils.logger import configure_logger
from mc_mace.utils.parse_input import (
    OUT_FILE_KEY,
    OUT_FOLDER_KEY,
    bar2eVA3,
    parse_yaml_input,
)


def create_out_files(sim_settings: dict[str, Any]) -> None:
    """
    Create necessary output files for the simulation, with backups if needed.

    Args:
        sim_settings (dict[str, Any]): Simulation settings parsed from the input YAML file.
    """
    for k in OUT_FILE_KEY:
        try:
            out_file = sim_settings["output files"][k]
            if out_file is not None:
                create_file_with_backup(out_file)
            else:
                logger.warning(f"missing `{k}` file name in input file")
        except KeyError:
            logger.debug(f"No `{k}` setting in input file")


def create_out_folders(sim_settings: dict[str, Any]) -> None:
    """
    Create necessary output folders for the simulation, with backups if needed.

    Args:
        sim_settings (dict[str, Any]): Simulation settings parsed from the input YAML file.
    """
    for k in OUT_FOLDER_KEY:
        try:
            out_folder = sim_settings[k]
            if out_folder is not None:
                create_folder_with_backup(out_folder)
            else:
                logger.warning(f"missing `{k}` folder name in input file")
        except KeyError:
            logger.debug(f"No `{k}` setting in input file")


def get_chem_pot(file_path: str, calculator: Calculator) -> float:
    """
    Compute the chemical potential from the energy of a system.

    Args:
        file_path (str): Path to the atomic configuration file.
        calculator (Calculator): ASE-compatible calculator for energy computation.

    Returns:
        float: Computed chemical potential in eV.
    """
    logger.debug(f"Computing the chemical potential from energy of system {file_path}")
    atoms = read(file_path)
    if not np.all(atoms.get_atomic_numbers() == atoms.get_atomic_numbers()[0]):
        logger.error(f"The system {file_path} contain more the one element")
    element = atoms.get_chemical_symbols()[0]
    atoms.calc = calculator
    mu = float(atoms.get_potential_energy() / len(atoms))
    logger.debug(f"mu({element}) = {mu:.3f} eV")
    return mu


def use_input_system_max_steps(system: Atoms, sim_settings: dict[str, Any]) -> dict[str, Any]:
    """
    Update the maximum displacement and volume change from the input system.

    Args:
        system (Atoms): ASE Atoms object representing the atomic configuration.
        sim_settings (dict[str, Any]): Simulation settings parsed from the input YAML file.

    Returns:
        dict[str, Any]: Updated simulation settings.
    """
    try:
        max_step = system.info["max_displacement"]
        logger.debug(f"Changing  'max displacement' {sim_settings['max displacement']} -> {max_step}")
        sim_settings["max displacement"] = float(max_step)
    except KeyError:
        logger.error("The info `max_displacement` no found")
        logger.warning(f"The input file `max_displacement` will be used ({sim_settings['max displacement']})")
        pass
    try:
        max_step = system.info["max_volume_change"] / system.get_volume()
        logger.debug(f"Changing  'max volume change' {sim_settings['max volume change']} -> {max_step}")
        sim_settings["max volume change"] = float(max_step)
    except KeyError:
        logger.error("The info `max_volume_change` no found")
        logger.warning(f"The input file `max_volume_change` will be used ({sim_settings['max volume change']})")
        pass
    return sim_settings


class BaseSimulation(ABC):
    """
    Abstract Base Class for Monte Carlo Simulations.

    This class provides the core setup and utility functions for Monte Carlo simulations, including
    input parsing, system initialization, calculator configuration, and chemical potential computation.

    Attributes:
        input_file (Path | str): Path to the YAML input file.
        log_file (str): Path to the log file.
        log_level (Path | str): Logging verbosity level.
        colorize (bool): Whether to enable colored logging.
        device (str): Computational device (e.g., "cuda" or "cpu").
        sim_settings (dict | None): Parsed simulation settings.
        system (Atoms | None): ASE Atoms object representing the simulation system.
        calculator (Calculator | None): ASE-compatible calculator for energy computations.
        engine (MC | None): Monte Carlo engine managing the simulation.
    """

    def __init__(self, input_file: Path | str, log_file: Path | str, log_level: str, colorize: bool, device: str):
        """
        Initialize the BaseSimulation.

        Args:
            input_file (Path | str): Path to the YAML input file.
            log_file (Path | str): Path to the log file.
            log_level (str): Logging verbosity level.
            colorize (bool): Enable colored logging.
            device (str): Computational device to use for calculations.
        """
        self.input_file = input_file
        self.log_file = log_file
        self.log_level = log_level
        self.colorize = colorize
        self.device = device
        self.sim_settings: dict[str, Any] | None = None
        self.system: Atoms | None = None
        self.calculator: Calculator | Any = None
        self.engine: MC | None = None

    def initialize(self) -> None:
        """
        Initialize the simulation.

        This includes configuring logging, parsing input files, creating output files and folders,
        and setting up the simulation system and calculator.
        """
        print_header()
        create_file_with_backup(self.log_file)
        with open(self.log_file, "a") as f_:
            print_header(f_)
        configure_logger(self.log_level.upper(), colorize=self.colorize, log_file=self.log_file)
        self.sim_settings = parse_yaml_input(self.input_file)
        create_out_files(self.sim_settings)
        create_out_folders(self.sim_settings)
        self._load_system()
        self._set_calculator()

    def _load_system(self) -> None:
        self.system = clean_ase_read(self.sim_settings["system"])  # type: ignore[index]
        if self.sim_settings["continue"]:  # type: ignore[index]
            logger.info("Continuing from input file")
            logger.info(f"Using the max step for position and volume from `{self.sim_settings['system']}` file")  # type: ignore[index]
            self.sim_settings = use_input_system_max_steps(self.system, self.sim_settings)  # type: ignore[arg-type]

    def _set_calculator(self) -> None:
        with warnings.catch_warnings():
            logger.debug("Loading MACE model")
            from mace.calculators import MACECalculator

            self.calculator = MACECalculator(model_paths=self.sim_settings["mace_model"], device=self.device)  # type: ignore[index]

    def _compute_chemical_potentials(self) -> tuple[list[str], list[float]]:
        elements = []
        potentials = []
        for element, value in self.sim_settings["chemical potential"].items():  # type: ignore[index]
            elements.append(element)
            if isinstance(value, str):
                mu = get_chem_pot(value, self.calculator)
                self.sim_settings["chemical potential"][element] = mu  # type: ignore[index]
                potentials.append(mu)
            else:
                potentials.append(value)
        return elements, potentials

    def _set_engine(self) -> None:
        elements, potentials = self._compute_chemical_potentials()
        if self.sim_settings is not None:
            self.engine = MC(
                self.system,
                self.calculator,
                mus=potentials,
                insert_elements=elements,
                T=self.sim_settings["temperature"],
                P=bar2eVA3(self.sim_settings["pressure"]),
                steps=self.sim_settings["steps"],
                random_number_gen=np.random.default_rng(seed=self.sim_settings["seed"]),
                cutoff=self.sim_settings["cutoff"],
                max_displacement=self.sim_settings["max displacement"],
                max_volume_change=float(self.sim_settings["max volume change"]) * self.system.get_volume(),  # type: ignore[union-attr]
                creation_max_attempts=self.sim_settings["max attempts"]["creation"],
                destruction_max_attempts=self.sim_settings["max attempts"]["destruction"],
                n_max=self.sim_settings["max atoms"],
                n_min=self.sim_settings["min atoms"],
            )
        else:
            raise RuntimeError("`sim_settings` attribute empty, run the `initialize()` method first")

    def _get_ensemble_settings(self) -> dict[str, Any]:
        if self.sim_settings is not None:
            simulation_settings = dict(
                engine=self.engine,
                steps=self.sim_settings["steps"],
                step_probability=self.sim_settings["probabilities"],
                random_number_gen=np.random.default_rng(seed=self.sim_settings["seed"]),
                out_thermo=self.sim_settings["output files"]["thermo"],
                out_trj=self.sim_settings["output files"]["trajectory"],
                out_events=self.sim_settings["output files"]["events"],
                out_restart=self.sim_settings["output files"]["restart"],
                out_state_folder=self.sim_settings["states folder"],
                save_thermo_step=self.sim_settings["saving step"]["thermo"],
                save_trj_step=self.sim_settings["saving step"]["trajectory"],
                save_events_step=self.sim_settings["saving step"]["events"],
                save_restart_step=self.sim_settings["saving step"]["restart"],
                save_state_step=self.sim_settings["saving step"]["states"],
                tunning_step=self.sim_settings["tuning every"],
            )
        else:
            raise RuntimeError("`sim_settings` attribute empty, run the `initialize()` method first")
        return simulation_settings

    @abstractmethod
    def run(self) -> None:
        """
        Abstract method to run the simulation.

        Subclasses must implement this method with the logic for executing the Monte Carlo simulation.
        """
        pass
