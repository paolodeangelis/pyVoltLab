from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# Define default values for optional parameters
DEFAULTS = {
    "optimizer": {
        "type": "FIRE2",
        "fmax": 0.05,
        "max steps": 10000,
    },
    "output files": {
        "thermo": "thermo.csv",
        "trajectory": "trj.xyz",
        "voltage": "voltage.csv",
        "convex hull": "convexhull.csv",
    },
    "states folder": "states",
}

# List of required parameters
REQUIRED_ENTRIES = ["system", "working ion", "mace_model"]


def parse_yaml_voltage_input(file_path: Path | str) -> dict[Any, Any]:
    """
    #TODO
    Parse a YAML input file, validate required parameters, and assign defaults
    to optional parameters if they are missing.

    Args:
        file_path (str): Path to the YAML input file.

    Returns:
        dict: Parsed and validated input data with optional parameters set to defaults.

    Raises:
        ValueError: If the YAML file is missing any required parameters or cannot be parsed.
    """
    try:
        # Load YAML file
        with open(file_path) as file:
            config = yaml.safe_load(file)
        logger.debug(f"Successfully loaded YAML file: {file_path}")

        # Check for required entries
        missing_entries = [entry for entry in REQUIRED_ENTRIES if entry not in config]
        if missing_entries:
            raise ValueError(f"Missing required parameters in chemical potential PID input file: {missing_entries}")

        logger.debug("All required parameters are present.")

        # Assign default values for optional parameters
        for key, default_value in DEFAULTS.items():
            if key not in config:
                config[key] = default_value
                logger.debug(f"Optional parameter '{key}' missing. Using default: {default_value}")
        return config  # type: ignore[no-any-return]

    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise ValueError(f"Input file not found: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {file_path}. Error: {e}")
        raise ValueError(f"Error parsing YAML file: {file_path}. Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e
