from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# Define default values for optional parameters
DEFAULTS_MACE = {
    "continue": False,
    "removal_method": "semi_brute_force",
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
REQUIRED_ENTRIES_MACE = ["system", "working ion", "mace_model"]

# Define default values for optional parameters
DEFAULTS_DFT = {
    "continue": False,
    "removal_method": "semi_brute_force",
    "calculation": "scf",
    "restart_mode": "from_scratch",
    "verbosity": "low",
    "outdir": "./tmp",
    "prefix": "pwscf",
    "max_seconds": 82800,  # 23 hours
    "tstress": True,
    "tprnfor": True,
    "etot_conv_thr": 1e-5,  # pwscf default = 1e-4
    "forc_conv_thr": 1e-4,  # pwscf default = 1e-3
    "input_dft": "pbe",
    "occupations": "smearing",
    "degauss": 0.01,
    "smearing": "cold",
    "conv_thr": 1e-8,  # pwscf default = 1e-6
    "electron_maxstep": 1000,
    "mixing_mode": "plain",
    "mixing_beta": 0.7,
    "diagonalization": "david",
    "startingwfc": "atomic",
    "koffset": [0, 0, 0],
    #
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
    "QE_dir": "QE",
}

# List of required parameters
REQUIRED_ENTRIES_DFT = [
    "working ion",
    "system",
    "ecutwfc",
    "ecutrho",
    "pseudopotentials",
    "command",
    "pseudo_dir",
    "kpts",
]


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

    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise ValueError(f"Input file not found: {file_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {file_path}. Error: {e}")
        raise ValueError(f"Error parsing YAML file: {file_path}. Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise e
    if "mace_model" in config:
        logger.debug("Detected MACE calculator.")
        REQUIRED_ENTRIES = REQUIRED_ENTRIES_MACE
        DEFAULTS = DEFAULTS_MACE
    elif "pseudopotentials" in config:
        logger.debug("Detected SCF Espresso calculator.")
        REQUIRED_ENTRIES = REQUIRED_ENTRIES_DFT
        DEFAULTS = DEFAULTS_DFT
    else:
        raise ValueError(
            "Only the MACE and SCF Espresso calculators are supported. Please verify your input file and ensure that all required parameters for the selected calculator are correctly provided."
        )

    # Check for required entries
    missing_entries = [entry for entry in REQUIRED_ENTRIES if entry not in config]
    if missing_entries:
        raise ValueError(f"Missing required parameters in file: {missing_entries}")

    logger.debug("All required parameters are present.")

    # Assign default values for optional parameters
    for key, default_value in DEFAULTS.items():
        if key not in config:
            config[key] = default_value
            logger.debug(f"Optional parameter '{key}' missing. Using default: {default_value}")
    return config  # type: ignore[no-any-return]
