import os
import shutil as sh
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from ase import Atoms
from ase.io import read
from loguru import logger


def save_dict_to_yaml(data: dict[str, Any], file_path: str | Path) -> None:
    """
    Save a dictionary to a YAML file.

    Args:
        data (Dict): The dictionary to be saved.
        file_path (str | Path): The path to the YAML file where the dictionary will be saved.

    Example:
        data = {
            "name": "mc_mace",
            "version": "1.0.0",
            "authors": ["Your Name <your.email@example.com>"],
        }
        save_dict_to_yaml(data, "output.yaml")
    """
    try:
        with open(file_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)
        logger.debug(f"Dictionary successfully saved to {file_path}.")
    except Exception as e:
        logger.error(f"Failed to save dictionary to {file_path}. Error: {e}")


def append_line_to_file(file_path: str | Path, lines: list[str] | str) -> None:
    """
    Append a line to a file.

    Args:
        file_path (str | Path): Path to the file.
        line (list): Line to append to the file.
    """
    if isinstance(lines, str):
        lines = [lines]
    try:
        with open(file_path, "a") as file:
            for line in lines:
                file.write(line + "\n")
    except Exception as e:
        logger.error(f"Failed to append line to file: {file_path}. Error: {e}")


def create_file_with_backup(file_path: str | Path) -> None:
    """
    Create an empty file. If the file already exists, create a backup and then create a new file.
    Creates the directory structure if it doesn't exist.

    Args:
        file_path (Union[str, Path]): Path to the file to be created.

    Raises:
        OSError: If there's an error creating directories, backup, or the new file.
    """
    path = Path(file_path)

    try:
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            # Create backup with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.parent / f"{path.name}.{timestamp}.bak"

            # Copy existing file to backup
            sh.copy2(path, backup_path)  # copy2 preserves metadata
            logger.warning(f"File {path} already exists")
            logger.warning(f"Backup created: {backup_path}")

        # Create new empty file
        with open(file_path, "w") as file:  # noqa: F841
            pass
        logger.debug(f"New file created: {path}")

    except OSError as e:
        logger.error(f"Failed to create file or backup for {path}. Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while creating {path}. Error: {str(e)}")
        raise


def create_folder_with_backup(folder_path: str | Path) -> None:
    """
    Create a folder. If the folder already exists, create a backup by renaming
    it with a timestamp and then create a new empty folder.

    Args:
        folder_path (str | Path): Path to the folder to be created.
    """
    try:
        if os.path.exists(folder_path):
            # Create a backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{folder_path}_{timestamp}.bak"
            sh.move(folder_path, backup_path)
            logger.warning(f"Folder {folder_path} already exist")
            logger.warning(f"Backup created: {backup_path}")

        # Create the new folder
        folder_path = Path(folder_path)
        # os.makedirs(folder_path, exist_ok=True)
        folder_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"New folder created: {folder_path}")
    except Exception as e:
        logger.error(f"Failed to create folder or backup: {folder_path}. Error: {e}")


def clean_ase_read(file_path: str | Path) -> Atoms:
    atoms = read(file_path)
    atoms_clean = Atoms(symbols=atoms.symbols, positions=atoms.positions, cell=atoms.cell, pbc=True)
    # additional inf
    for k_info in [
        "volume",
        "potential_energy",
        "energy",
        "success_step",
        "step",
        "max_displacement",
        "max_volume_change",
    ]:
        try:
            atoms_clean.info[k_info] = atoms.info[k_info]
        except KeyError:
            pass

    return atoms_clean
