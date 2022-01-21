"""
File holding the base class for all SbSOvRL classes which are needed for the command line based application of this toolbox.
"""
import pathlib
from typing import Optional, List, Any
from pydantic import BaseModel, Extra
from SbSOvRL.util.logger import logging
import datetime
from collections import OrderedDict

def add_save_location_if_elem_is_o_dict(possible_value: Any, save_location: pathlib.Path):
    """This function adds the save_location to all ordered dicts in possible_value if it is not already present. This function is used to forward the save_location variable to all child objects so that all objects can access it.

    Also handles adding the save_location to objects inside of lists (these can even be multiple lists deep)

    Args:
        possible_value (Any): Variable holding potential objects which need the save_location. 
        save_location (str): string that defines the location
    """
    if isinstance(possible_value, OrderedDict):
        if "save_location" not in possible_value:
            possible_value["save_location"] = save_location
    elif isinstance(possible_value, list):
        for item in possible_value:
            add_save_location_if_elem_is_o_dict(item, save_location)

class SbSOvRL_BaseModel(BaseModel):
    """
    Base class for all SbSOvRL classes which are needed for the command line based application of this toolbox.
    """
    save_location: pathlib.Path #: Definition of the save location of the logs and validation results. Should be given as a standard string will be preconverted into a pathlib.Path. If {} are present in the string the current timestamp is added.
    logger_name: Optional[str] = None   #: name of the logger. If this variable gives you trouble, the framework is at least a little bit buggy. Does not need to be set. And might be changed if set by user, if multi-environment training is utilized.

    # private fields
    # _verbosity: Any = PrivateAttr(default=None) #: Defining the verbosity of the training process and environment loading

    def __init__(self, **data: Any) -> None:
        if "save_location" in data:
            if type(data["save_location"]) is str:  # This is so that the save location always gets the same 
                data["save_location"] = SbSOvRL_BaseModel.convert_path_to_pathlib_and_add_datetime_if_applicable(data["save_location"])
            # if a save_location is present in the current object definition add this save_location also to all object definition which are direct dependends
            for _, value in data.items():
                add_save_location_if_elem_is_o_dict(value, data["save_location"])
        super().__init__(**data)


    @classmethod
    def convert_path_to_pathlib_and_add_datetime_if_applicable(cls, v:str):
        """Adds a datetime timestamp to the save_location if {} present. This is done to make it easier to differentiate different runs without the need to change the base name with every run.

        This function also ensures that the directory is created.

        Args:
            v ([type]): Value to validate

        Returns:
            pathlib.Path: Path like object. If applicable with timestamp. 
        """
        path = pathlib.Path(v.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _check_list(self, list_item: List[Any], logger_name: str):
        """Recursivly goes through lists and add the logger_name where applicable.

        Args:
            list_item (List[Any]): list in which to check if items reside which need checking for logger names
            logger_name (str): logger name to add to all applicable objects
        """
        for item in list_item:
            if isinstance(item, list):
                self._check_list(item, logger_name)
            elif isinstance(item, SbSOvRL_BaseModel):
                item.set_logger_name_recursively(logger_name)

    def set_logger_name_recursively(self, logger_name: str):
        """Sets the given logger_name for the current object and all attributes which have the ``set_logger_recursively`` method.

        Note: 
            Please note: It is the callings functions responsibility to ensure that the logger actually exists.

        Args:
            logger_name (str): Name of the logger to set.
        """
        self.logger_name = logger_name
        for _, value in self.__dict__.items():
            if isinstance(value, list):
                self._check_list(value, logger_name)
            else:
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