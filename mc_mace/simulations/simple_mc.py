from pathlib import Path

from loguru import logger

from mc_mace.ensembles import NPT, muPT
from mc_mace.utils.io import save_dict_to_yaml

from .simulation_abc import BaseSimulation


class Simulation(BaseSimulation):
    """
    Monte Carlo Simulation Runner.

    This class extends `BaseSimulation` and provides functionality for running Monte Carlo simulations
    using the NPT or µPT ensembles. It handles configuration, engine setup, and simulation execution.

    Attributes:
        ensemble (NPT | muPT | None): The selected ensemble for the simulation.
    """

    def __init__(self, input_file: Path | str, log_file: Path | str, log_level: str, colorize: bool, device: str):
        """
        Initialize the Simulation runner.

        Args:
            input_file (Path | str): Path to the YAML input file containing simulation settings.
            log_file (Path | str): Path to the log file.
            log_level (Path | str): Logging verbosity level (e.g., "DEBUG", "INFO").
            colorize (bool): Whether to enable colored logging output.
            device (str): Computational device to use (e.g., "cuda" or "cpu").
        """
        super().__init__(input_file, log_file, log_level, colorize, device)
        self.ensemble = None

    def run(self) -> None:
        """
        Execute the Monte Carlo simulation.

        This method initializes the simulation, sets up the Monte Carlo engine, configures the ensemble,
        and runs the simulation loop.

        Raises:
            ValueError: If the ensemble type specified in the input settings is unsupported.
        """
        self.initialize()
        self._set_engine()
        ensemble_settings = self._get_ensemble_settings()
        if self.sim_settings["ensemble"].lower() == "npt":  # type: ignore[index]
            self.ensemble = NPT(**ensemble_settings)  # type: ignore[assignment]
        elif self.sim_settings["ensemble"].lower() == "mupt":  # type: ignore[index]
            self.ensemble = muPT(**ensemble_settings)  # type: ignore[assignment]
        else:
            logger.error("Unsupported ensemble type")
            raise ValueError("Unsupported ensemble type")
        save_dict_to_yaml(self.sim_settings, "full_simulation_settings.yaml")  # type: ignore[arg-type]
        self.ensemble.run()  # type: ignore[attr-defined]
