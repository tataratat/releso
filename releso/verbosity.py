"""File hold definition for verbosity settings.

File holding the class defining the verbosity of the problem and defining the
loggers.
"""

import datetime
import pathlib
from typing import Any, Literal

from pydantic import validator
from pydantic.fields import PrivateAttr

from releso.base_model import BaseModel
from releso.util.logger import VerbosityLevel, logging, set_up_logger


class Verbosity(BaseModel):
    """Verbosity class.

    Defines the settings for the different loggers used in the current
    experiment. This class is the only class which is copied to all
    children. (this happens outside of the the standard channels and will
    hopefully not break with multiprocessing)

    Please note, the parser logger only ever can have the following name
    ``ReLeSO_parser``
    """

    #: VerbosityLevel of the parser logger. This logger should only generate
    #: messages during the setup of the experiment.
    parser: Literal["ERROR", "WARNING", "DEBUG", "INFO"] = "INFO"
    #: VerbosityLevel of the environment parsers. These loggers will generate
    #: messages during the execution of the experiments.
    #: (Training and Validation)
    environment: Literal["ERROR", "WARNING", "DEBUG", "INFO"] = "INFO"
    #: Path where the log files should be saved to. Will be inside the
    #: base_save_location.
    logfile_location: str = "logging/"
    #: Whether or not to also print the log messages to the console.
    console_logging: bool = False
    #: Base name of all logger. Defaults to "ReLeSO"
    base_logger_name: str = "ReLeSO"
    #: Name extensions for all environment logger. Defaults to "environment"
    environment_extension: str = "environment"

    # private fields
    #: save different loggers for the different instances
    _environment_logger: str = PrivateAttr(default="")
    #: save different loggers for the different instances
    _environment_validation_logger: str = PrivateAttr(default="")

    @validator("parser", "environment", always=True)
    @classmethod
    def convert_literal_str_to_verbosity_level(cls, v):
        """Validator for parser and environment variable.

        Validation function converting the string representation of the enum
        to the correct enum item.

        Args:
            v (str)): String representation of the VerbosityLevel

        Returns:
            VerbosityLevel: Enum object of the wanted value
        """
        if v == "ERROR":
            return VerbosityLevel.ERROR
        elif v == "WARNING":
            return VerbosityLevel.WARNING
        elif v == "DEBUG":
            return VerbosityLevel.DEBUG
        elif v == "INFO":
            return VerbosityLevel.INFO

    @validator("logfile_location", always=True)
    @classmethod
    def make_logfile_location_absolute(cls, v, values):
        """Validation function resolves and makes the log path absolute.

        Also adds the current timestamp to the log path if a {} is present in
        the given path.

        Args:
            v (str): [description]
            values ([type]): [description]

        Returns:
            pathlib.Path:
                pathlib representation of the path with if applicable the
                current timestamp
        """
        path: pathlib.Path = None
        v = v.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if "save_location" in values:
            # the path where the logger should write to is the
            # save_location/logfile_location
            path = values["save_location"] / v
        else:
            # the path where the logger should write to is the
            # calling_folder/logfile_location
            path = pathlib.Path(v).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def __init__(self, **data: Any) -> None:
        """Constructor verbosity parser."""
        super().__init__(**data)
        # create parser logger
        parser_name = "releso_parser"

        parser_logger = set_up_logger(
            parser_name,
            self.logfile_location,
            self.parser,
            self.console_logging,
        )

        # create base rl logger
        self._environment_logger = "_".join(
            filter(
                ("").__ne__,
                [self.base_logger_name, self.environment_extension],
            )
        )
        self.add_environment_logger_with_name_extension("")
        # create base rl logger for validation environment
        self._environment_validation_logger = "_".join(
            filter(
                ("").__ne__,
                [
                    self.base_logger_name,
                    self.environment_extension,
                    "validation",
                ],
            )
        )
        self.add_environment_logger_with_name_extension("validation")

        parser_logger.debug(
            f"Setup logger. Parser logger has logging level: {self.parser}; "
            f"Environment logger has logging level: {self.environment}"
        )

    def add_environment_logger_with_name_extension(
        self, extension: str
    ) -> logging.Logger:
        """Initializes a logger with the settings for the environment logger.

        The name of the base environment logger is extended by the
        :attr:`extension`. The name if the created logger is found by joining
        the strings of the following variables by an underscore.
        :attr:`Verbosity.base_logger_name`,
        :attr:`Verbosity.environment_extension`, :attr:`extension` Any empty
        strings are ignored.

        Args:
            extension (str):
                Used to differentiate between different environment loggers.
                If empty the base logger is created/returned.

        Returns:
            logging.Logger:
                Logger of the name defined name. The name definition is
                explained in the main documentation of the function.
        """
        logger_name = "_".join(
            filter(
                ("").__ne__,
                [self.base_logger_name, self.environment_extension, extension],
            )
        )
        if not logging.getLogger(logger_name).hasHandlers() or not (
            len(logging.getLogger(logger_name).handlers) > 0
        ):
            logger = set_up_logger(
                logger_name,
                self.logfile_location,
                self.environment,
                self.console_logging,
            )
        else:
            logger = logging.getLogger(logger_name)
        return logger
