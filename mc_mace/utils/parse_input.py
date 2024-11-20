from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from scipy.constants import elementary_charge

OUT_FILE_KEY = ["trajectory", "thermo", "events"]
OUT_FOLDER_KEY = ["states folder"]

# Define default values for optional parameters
DEFAULTS = {
    "npt": {
        "temperature": 300.0,  # Kelvin
        "pressure": 1.0,  # Bar
        "cutoff": 5.0,  # Angstroms
        "chemical potential": {},
        "seed": None,
        "max displacement": 0.15,
        "max volume change": 0.01,
        "probabilities": {"position": 0.75, "volume": 0.25},
        "continue": False,
        "tuning every": 10,
    },
    "mupt": {
        "temperature": 300.0,  # Kelvin
        "pressure": 1.0,  # Bar
        "cutoff": 5.0,  # Angstroms
        "seed": None,
        "max displacement": 0.15,
        "max volume change": 0.01,
        "probabilities": {
            "position": 0.75,
            "volume": 0.23,
            "creation": 0.01,
            "destruction": 0.01,
        },
        "continue": False,
        "tuning every": 10,
    },
}
DEFAULTS_OUT = {
    "npt": {
        "restart": "restart.xyz",
    },
    "mupt": {
        "restart": "restart.xyz",
    },
}
DEFAULTS_STEP = {
    "npt": {
        "restart": 10,
    },
    "mupt": {
        "restart": "restart.xyz",
    },
}
# List of required parameters
REQUIRED_ENTRIES = {
    "npt": ["ensemble", "mace_model", "system", "steps"],
    "mupt": ["chemical potential"],
}


def convert_pressure(inputs: dict[str, Any]) -> dict[str, Any]:
    """Convert pressure from Bar to eV/A^3

    Args:
        inputs (dict): inputs dictionary
    """
    try:
        inputs["pressure"] = inputs["pressure"] * 1e5 / elementary_charge * 1e-30
    except KeyError:
        pass
    return inputs


def bar2eVA3(pressure: float) -> float:
    """Convert pressure from Bar to eV/A^3

    Args:
        pressure (float): Pressure in Bar

    Returns:
        float: Pressure in eV/A^3.
    """
    return float(pressure * 1e5 / elementary_charge * 1e-30)


def parse_yaml_input(file_path: Path | str) -> dict[Any, Any]:
    """
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
        missing_entries = [entry for entry in REQUIRED_ENTRIES[config["ensemble"].lower()] if entry not in config]
        if missing_entries:
            raise ValueError(f"Missing required parameters in input file: {missing_entries}")

        logger.debug("All required parameters are present.")

        # Assign default values for optional parameters
        for key, default_value in DEFAULTS[config["ensemble"].lower()].items():  # type: ignore[attr-defined]
            if key not in config:
                config[key] = default_value
                logger.debug(f"Optional parameter '{key}' missing. Using default: {default_value}")
        for key, default_value in DEFAULTS_OUT[config["ensemble"].lower()].items():
            if key not in config["output files"]:
                config["output files"][key] = default_value
                logger.debug(f"Optional output file '{key}' missing. Using default: {default_value}")
        for key, default_value in DEFAULTS_STEP[config["ensemble"].lower()].items():  # type: ignore[attr-defined]
            if key not in config["saving step"]:
                config["saving step"][key] = default_value
                logger.debug(f"Optional output saving step '{key}' missing. Using default: {default_value}")
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
