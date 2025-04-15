"""ReLeSO logging.

This file holds the verbosity definitions and helper functions for the logger
used in this package.
"""

import enum
import logging
import pathlib
from typing import Optional


def get_parser_logger() -> logging.Logger:
    """Get the logger which is to be used during the parsing of the json file.

    Returns:
        logging.Logger: requested logger
    """
    return logging.getLogger("ReLeSO_parser")


class VerbosityLevel(enum.IntEnum):
    """Defines the verbosity level of the logger."""

    #: Writes only errors to the log
    ERROR = logging.ERROR
    #: Writes errors and warning to the log
    WARNING = logging.WARNING
    #: Writes errors, warnings and info fields to the log
    INFO = logging.INFO
    #: Writes everything to the log.
    DEBUG = logging.DEBUG


def set_up_logger(
    logger_name: str = "",
    log_file_location: pathlib.Path = pathlib.Path("."),
    verbosity: VerbosityLevel = VerbosityLevel.INFO,
    console_logging: bool = False,
    logger: Optional[logging.Logger] = None,
) -> logging.Logger:
    """Create a logger instance with a specified name.

    Author:
        Daniel Wolff (wolff@cats.rwth-aachen.de),
        Clemens Fricke (clemens.david.fricke@tuwien.ac.at)

    Args:
        logger_name (str, optional):
            Name of the logging instance. Defaults to ''.
        log_file_location (Path):
            Path to the directory into which the log file(s) will be placed
            into. Defaults to ".".
        verbosity (VerbosityLevel):
            Enum value for the verbosity of the given logger.
            Defaults to VerbosityLevel.INFO
        console_logging (bool):
            Toggle whether to also log into the console. Defaults to False.
        logger (Optional[logging.Logger]):
            Will add handles defined in this function to the given logger.
            Instead of creating a new logger instance. Defaults to None.

    Returns:
        logging.Logger:
            Configured logger instance for simultaneously writing to file and
            console
    """
    if not logger:  # if no logger is given a new logger is created
        logger = logging.getLogger(logger_name)
    # with existing loggers the wanted logger name is added to the existing
    # name. (multiprocessing)
    else:
        logger.name += f"_{logger_name}"
    # delete old handlers
    logger.handlers.clear()
    # log everything which is debug or above
    logger.setLevel(verbosity.value)
    # create formatter for file output and add it to the handlers
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_file_location.mkdir(parents=True, exist_ok=True)

    if verbosity <= VerbosityLevel.WARNING:
        log_file_name = log_file_location / f"{logger_name}.log"
        # create log handler which logs even debug messages
        lh = logging.FileHandler(log_file_name)
        # create console handler with a higher log level
        lh.setLevel(verbosity.value)
        lh.setFormatter(file_formatter)
        logger.addHandler(lh)

    if console_logging:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter for console output and add it to the handlers
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

    # create error handler which logs only error messages
    err_file_name = log_file_location / f"{logger_name}.err"
    eh = logging.FileHandler(err_file_name)
    eh.setLevel(logging.ERROR)
    eh.setFormatter(file_formatter)
    # add the handlers to logger
    logger.addHandler(eh)

    return logger
