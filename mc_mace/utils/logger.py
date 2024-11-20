import sys
from typing import Any

from loguru import logger
from tqdm import tqdm


def logger_formatter(record: dict[str, Any]) -> str:
    """Format the logger output.

    Args:
        record (dict): The logging record to be formatted.

    Returns:
        str: Formatted logging message.
    """
    if record["level"].no <= 30:
        return (
            "[ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: ^8}</level> ] <level>{message}</level>\n{exception}"
        )
    else:
        return (
            "[ <green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: ^8}</level> > <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> ] "
            "<level>{message}</level>\n{exception}"
        )


def configure_logger(log_level: str, log_file: str = "log.log", colorize: bool = True) -> None:
    """Configure the Loguru logger to log both to console and a file.

    Args:
        log_level (str): Logging level to set for the logger (e.g., "DEBUG", "INFO").
        log_file (str, optional): Path to the log file. Defaults to "app.log".
        colorize (bool, optional): Whether to enable colorized output for the console. Defaults to True.
    """
    logger.remove()  # Remove default logger

    # Add console logging
    try:
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            level=log_level.upper(),
            colorize=colorize,
            format=logger_formatter,
        )
    except ModuleNotFoundError:
        logger.add(
            sys.stdout,
            level=log_level.upper(),
            colorize=True,
            format=logger_formatter,
        )
    # Add file logging
    logger.add(
        log_file,
        level="INFO",
        format=logger_formatter,
        rotation="10 MB",  # Rotate log file after reaching 10 MB
        retention="10 days",  # Retain rotated logs for 10 days
        compression="zip",  # Compress rotated log files
    )
