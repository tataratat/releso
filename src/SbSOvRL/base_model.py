"""
File holding the base class for all SbSOvRL classes which are needed for the command line based application of this toolbox.
"""
import pathlib
from typing import Optional
from pydantic import BaseModel, Extra
from pydantic.class_validators import validator
from typing import Any
from SbSOvRL.util.logger import logging
import datetime

class SbSOvRL_BaseModel(BaseModel):
    """
    Base class for all SbSOvRL classes which are needed for the command line based application of this toolbox.
    """
    save_location: str    #: Definition of the save location of the logs and validation results
    logger_name: Optional[str] = None   #: name of the logger. If this variable gives you trouble, the framework is at least a little bit buggy. Does not need to be set. And might be changed if set by user, if multi-environment training is utilized.

    # private fields
    # _verbosity: Any = PrivateAttr(default=None) #: Defining the verbosity of the training process and environment loading

    def __init__(__pydantic_self__, **data: Any) -> None:
        if "save_location" in data:
            # if a save_location is present in the current object definition add this save_location also to all object definition which are direct dependends
            for _, value in data.items():
                if type(value) is dict and "save_location" not in value:
                    value["save_location"] = data["save_location"]
        super().__init__(**data)

    @validator("save_location")
    @classmethod
    def convert_path_to_pathlib_and_add_datetime_if_applicable(cls, v):
        """Adds a datetime timestamp to the save_location if \{\} present. This is done to make it easier to differentiate different runs without the need to change the base name with every run.

        This function also ensures that the directory is created.

        Args:
            v ([type]): Value to validate

        Returns:
            pathlib.Path: Path like object. If applicable with timestamp. 
        """
        path = pathlib.Path(v.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def set_logger_name_recursively(self, logger_name: str):
        """Sets the given logger_name for the current object and all attributes which have the ``set_logger_recursively`` method.

        Note: 
            Please note: It is the callings functions responsibility to ensure that the logger actually exists.

        Args:
            logger_name (str): Name of the logger to set.
        """
        self.logger_name = logger_name
        for _, value in self.__dict__.items():
            attr = getattr(value, "set_logger_name_recursively", None)
            if callable(attr):
                attr(logger_name)

    def get_logger(self) -> logging.Logger:
        """Gets the currently defined environment logger.

        Returns:
            logging.Logger: logger which is currently to be used.
        """
        return logging.getLogger(self.logger_name)

    class Config:
        extra = Extra.forbid