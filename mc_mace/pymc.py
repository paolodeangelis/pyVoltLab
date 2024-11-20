"""Provide the primary functions."""

import warnings
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read
from loguru import logger

from mc_mace.ensembles import NPT, muPT
from mc_mace.mc import MC
from mc_mace.utils.header import print_header
from mc_mace.utils.io import (
    clean_ase_read,
    create_file_with_backup,
    create_folder_with_backup,
    save_dict_to_yaml,
)
from mc_mace.utils.logger import configure_logger
from mc_mace.utils.parse_input import (
    OUT_FILE_KEY,
    OUT_FOLDER_KEY,
    bar2eVA3,
    parse_yaml_input,
)

app = typer.Typer(help="Monte Carlo simulation with MACE CLI.")


def create_out_files(sim_settings: dict[str, Any]) -> None:
    for k in OUT_FILE_KEY:
        try:
            out_file = sim_settings["output files"][k]
            create_file_with_backup(out_file)
        except KeyError:
            logger.debug(f"No `{k}` setting in input file")


def create_out_folders(sim_settings: dict[str, Any]) -> None:
    for k in OUT_FOLDER_KEY:
        try:
            out_file = sim_settings[k]
            create_folder_with_backup(out_file)
        except KeyError:
            logger.debug(f"No `{k}` setting in input file")


def get_chem_pot(file_path: str, calculator: Calculator) -> float:
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


@app.command(help="Run MC simulation")  # type: ignore[misc]
def run(
    input_file: Annotated[Path, typer.Argument(help="YAML input file ")],
    device: Annotated[str, typer.Option(help="Device for potential energy calculation (cuda, cpu)")] = "cuda",
    log_level: Annotated[str, typer.Option(help="Logger level (DEBUG, INFO, WARNING, ERROR)")] = "INFO",
    log_file: Annotated[str, typer.Option(help="Log file path")] = "simulation.log",
    colorize: Annotated[bool, typer.Option(help="Colorize colors")] = True,
) -> None:
    """Test cmd"""
    # warm up
    print_header()
    create_file_with_backup(log_file)
    with open(log_file, "a") as f_:
        print_header(f_)
    configure_logger(log_level.upper(), colorize=colorize, log_file=log_file)
    sim_settings = parse_yaml_input(input_file)
    create_out_files(sim_settings)
    create_out_folders(sim_settings)
    # load system
    system = clean_ase_read(sim_settings["system"])
    if sim_settings["continue"]:
        logger.info("Continuing option in input file")
        logger.info(f"Using the max step for position and volume from `{sim_settings['system']}` file")
        sim_settings = use_input_system_max_steps(system, sim_settings)
    # set calculator
    with warnings.catch_warnings():
        from mace.calculators import MACECalculator
    calculator = MACECalculator(model_paths=sim_settings["mace_model"], device=device)
    # chemical potential
    adding_atoms_elements = []
    adding_atoms_mu = []
    for k, v in sim_settings["chemical potential"].items():
        adding_atoms_elements.append(k)
        if isinstance(v, str):
            mu = get_chem_pot(v, calculator)
            sim_settings["chemical potential"][k] = mu
            adding_atoms_mu.append(mu)
        else:
            adding_atoms_mu.append(v)
    # set MC engine
    engine = MC(
        system,
        calculator,
        mus=adding_atoms_mu,
        insert_elements=adding_atoms_elements,
        T=sim_settings["temperature"],
        P=bar2eVA3(sim_settings["pressure"]),
        steps=sim_settings["steps"],
        random_number_gen=np.random.default_rng(seed=sim_settings["seed"]),
        cutoff=sim_settings["cutoff"],
        max_displacement=sim_settings["max displacement"],
        max_volume_change=float(sim_settings["max volume change"]) * system.get_volume(),
    )
    # set simulation
    simulation_settings = dict(
        engine=engine,
        steps=sim_settings["steps"],
        step_probability=sim_settings["probabilities"],
        random_number_gen=np.random.default_rng(seed=sim_settings["seed"]),
        out_thermo=sim_settings["output files"]["thermo"],
        out_trj=sim_settings["output files"]["trajectory"],
        out_events=sim_settings["output files"]["events"],
        out_restart=sim_settings["output files"]["restart"],
        out_state_folder=sim_settings["states folder"],
        save_thermo_step=sim_settings["saving step"]["thermo"],
        save_trj_step=sim_settings["saving step"]["trajectory"],
        save_events_step=sim_settings["saving step"]["events"],
        save_restart_step=sim_settings["saving step"]["restart"],
        save_state_step=sim_settings["saving step"]["states"],
        tunning_step=sim_settings["tuning every"],
    )

    if sim_settings["ensemble"].lower() == "npt":
        simulation = NPT(**simulation_settings)
    elif sim_settings["ensemble"].lower() == "mupt":
        simulation = muPT(**simulation_settings)  # type: ignore[assignment]
    save_dict_to_yaml(sim_settings, "full_simulation_settings.yaml")
    simulation.run()


# @app.command()
# def test2():
#     """Test cmd"""
#     log_level = "INFO"
#     configure_logger(log_level, colorize=True)
#     logger.debug("Test DEBUG")
#     logger.info("Test DEBUG")
#     logger.warning("Test WARNING")
#     logger.error("Test ERROR")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
