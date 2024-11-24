"""Provide the primary functions."""

from pathlib import Path
from typing import Annotated

import typer

from mc_mace.simulations import PIDTuning, Simulation

app = typer.Typer(help="Monte Carlo simulation with MACE CLI.")


@app.command(help="Run a Monte Carlo (MC) simulation.")  # type: ignore[misc]
def run(
    input_file: Annotated[Path, typer.Argument(help="YAML input file ")],
    device: Annotated[str, typer.Option(help="Device for potential energy calculation (cuda, cpu)")] = "cuda",
    log_level: Annotated[str, typer.Option(help="Logger level (DEBUG, INFO, WARNING, ERROR)")] = "INFO",
    log_file: Annotated[str, typer.Option(help="Log file path")] = "simulation.log",
    colorize: Annotated[bool, typer.Option(help="Colorize colors")] = True,
) -> None:
    """
    Run a Monte Carlo (MC) simulation.

    This command executes a Monte Carlo simulation based on the settings provided in the input YAML file.
    The simulation uses the MACE framework for potential energy calculations.

    Example:
        python script.py run simulation_input.yaml --device cuda --log-level DEBUG

    Args:
        input_file (Path): Path to the YAML file containing simulation parameters.
        device (str): Computational device to use ('cuda' for GPU or 'cpu').
        log_level (str): Logging verbosity level (default: INFO).
        log_file (str): Path to the log file (default: simulation.log).
        colorize (bool): Whether to colorize log output (default: True).
    """
    simulation = Simulation(input_file, log_file=log_file, log_level=log_level, colorize=colorize, device=device)
    simulation.run()


@app.command(help="Perform dynamic PID tuning for chemical potential during an MC simulation.")  # type: ignore[misc]
def chem_pid(
    input_mc_file: Annotated[Path, typer.Argument(help="YAML input MC step file")],
    input_pid_file: Annotated[Path, typer.Argument(help="YAML input PID tuning file")],
    device: Annotated[str, typer.Option(help="Device for potential energy calculation (cuda, cpu)")] = "cuda",
    log_level: Annotated[str, typer.Option(help="Logger level (DEBUG, INFO, WARNING, ERROR)")] = "INFO",
    log_file: Annotated[str, typer.Option(help="Log file path")] = "simulation.log",
    colorize: Annotated[bool, typer.Option(help="Colorize colors")] = True,
) -> None:
    """
    Perform dynamic PID tuning for chemical potential.

    This command adjusts the chemical potential dynamically using a PID controller during a Monte Carlo simulation.
    The simulation is configured using two YAML files: one for the MC settings and another for PID tuning parameters.

    Example:
        python script.py chem-pid mc_input.yaml pid_input.yaml --device cuda --log-level DEBUG

    Args:
        input_mc_file (Path): Path to the YAML file containing Monte Carlo step parameters.
        input_pid_file (Path): Path to the YAML file containing PID tuning parameters.
        device (str): Computational device to use ('cuda' for GPU or 'cpu').
        log_level (str): Logging verbosity level (default: INFO).
        log_file (str): Path to the log file (default: simulation.log).
        colorize (bool): Whether to colorize log output (default: True).
    """
    simulation = PIDTuning(
        input_mc_file, pid_file=input_pid_file, log_file=log_file, log_level=log_level, colorize=colorize, device=device
    )
    simulation.run()


def main() -> None:
    """
    Entry point for the Monte Carlo simulation CLI.

    Use this command-line tool to run Monte Carlo simulations or dynamic PID tuning
    with customizable parameters and logging options.
    """
    app()


if __name__ == "__main__":
    """
    Main executable entry point.

    This allows the script to be run directly, providing a user-friendly CLI
    for Monte Carlo simulations and PID tuning.
    """
    main()
