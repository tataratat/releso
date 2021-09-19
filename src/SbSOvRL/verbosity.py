from pydantic import validator, root_validator
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.util.logger import VerbosityLevel, set_up_logger
import pathlib
from typing import Any, Literal
import datetime


class Verbosity(SbSOvRL_BaseModel): 
    parser: Literal["ERROR", "WARNING", "DEBUG", "INFO"] = "INFO"
    environment: Literal["ERROR", "WARNING", "DEBUG", "INFO"] = "INFO"
    SbSOvRL_logfile_location: str = "training_{}/logging/"
    console_logging: bool = False
    
    @validator("parser", "environment", always=True)
    @classmethod
    def convert_literal_str_to_verbosityLevel(cls, v):
        if v == "ERROR":
            return VerbosityLevel.ERROR
        elif v == "WARNING":
            return VerbosityLevel.WARNING
        elif v == "DEBUG":
            return VerbosityLevel.DEBUG
        elif v == "INFO":
            return VerbosityLevel.INFO

    @validator("SbSOvRL_logfile_location", always=True)
    @classmethod
    def make_logfile_location_absolute(cls, v):
        return pathlib.Path(v.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))).expanduser().resolve()
    
    # @root_validator()
    # @classmethod
    # def add(cls, values):
    #     """Adds a timestamp to the log file location so that multiple runs can be more easily be distinguished.
    #     """
    #     add_time_stamp = values["add_time_stamp"]
    #     if add_time_stamp:
    #         path = values["SbSOvRL_logfile_location"]
    #         path = path / 
    #         values["SbSOvRL_logfile_location"] = path
    #     return values

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # create parser logger
        parser_logger = set_up_logger("SbSOvRL_parser", self.SbSOvRL_logfile_location, self.parser, self.console_logging)

        # create environment logger
        set_up_logger("SbSOvRL_environment", self.SbSOvRL_logfile_location, self.environment, self.console_logging)

        parser_logger.info(f"Setup logger. Parser logger has logging level: {self.parser}; Environment logger has logging level: {self.environment}")

