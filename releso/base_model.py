"""Definition of the ReLeSO base Model.

File holding the base class for all ReLeSO classes which are needed for the
command line based application of this toolbox.
"""

import multiprocessing
import pathlib
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import pydantic

from releso.util.logger import logging
from releso.util.util_funcs import get_path_extension


def add_save_location_if_elem_is_o_dict(
    possible_value: Any, save_location: pathlib.Path
):
    """Add the save_location keyword to the element of it is a dict.

    This function adds the save_location to all ordered dicts in possible_value
    if it is not already present. This function is used to forward the
    save_location variable to all child objects so that all objects can access
    it.

    Also handles adding the save_location to objects inside of lists (these can
    even be multiple lists deep)

    Args:
        possible_value (Any): Variable holding potential objects which need
        the save_location.
        save_location (str): string that defines the location
    """
    if isinstance(possible_value, OrderedDict) or isinstance(
        possible_value, Dict
    ):
        if "save_location" not in possible_value:
            possible_value["save_location"] = save_location
    elif isinstance(possible_value, list):
        for item in possible_value:
            add_save_location_if_elem_is_o_dict(item, save_location)
    else:
        pass


class BaseModel(pydantic.BaseModel):
    """Base of all ReLeSO objects used for parsing.

    Base class for all ReLeSO classes which are needed for the command line
    based application of this toolbox.
    """

    #: Definition of the save location of the
    #: logs and validation results. Should be given as a standard string will
    #: be pre-converted into a pathlib.Path. If {} is present in the string the
    #: current timestamp is added if in a slurm job SLURM_JOB_ID are added.
    save_location: pathlib.Path
    #: name of the logger. If this variable gives you trouble, the framework is
    #: at least a little bit buggy. Does not need to be set. And might be
    #: changed if set by user, if multi-environment training is utilized.
    logger_name: Optional[str] = None

    def __init__(self, **data: Any) -> None:
        """Constructor for the ReLeSO basemodel object."""
        if "save_location" in data:
            # This is so that the save location always gets the same
            if isinstance(data["save_location"], str):
                data["save_location"] = (
                    BaseModel.convert_to_pathlib_add_datetime(
                        data["save_location"]
                    )
                )
            # if a save_location is present in the current object definition
            #  add this save_location also to all object definition which are
            #  direct dependents
            for value in data.values():
                add_save_location_if_elem_is_o_dict(
                    value, data["save_location"]
                )
        super().__init__(**data)

    @classmethod
    def convert_to_pathlib_add_datetime(cls, v: str):
        """Add timestamp to save_location, of applicable.

        Adds a datetime timestamp to the save_location if {} present and
        current slurm job id if available. Task arrays will be added to the
        same folder and each task gets its own subfolder. This is done to make
        it easier to differentiate different runs without the need to change
        the base name with every run and to easily group similar runs.

        This function also ensures that the directory is/directories are
        created.

        Args:
            v ([type]): Value to validate

        Returns:
            pathlib.Path: Path like object. If applicable with identifications.
        """
        path = (
            pathlib.Path(v.format(get_path_extension())).expanduser().resolve()
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _check_list(self, list_item: List[Any], logger_name: str):
        """Helper function for set_logger_name_recursively.

        Recursively goes through lists and add the logger_name where
        applicable.

        Args:
            list_item (List[Any]): list in which to check if items reside
            which need checking for logger names
            logger_name (str): logger name to add to all applicable objects
        """
        for item in list_item:
            if isinstance(item, list):
                self._check_list(item, logger_name)
            elif isinstance(item, BaseModel):
                item.set_logger_name_recursively(logger_name)

    def set_logger_name_recursively(self, logger_name: str):
        """Set the logger_name variable for all child elements.

        Sets the given logger_name for the current object and all attributes
        which have the ``set_logger_recursively`` method.

        Note:
            Please note: It is the callings functions responsibility to ensure
            that the logger actually exists.

        Args:
            logger_name (str): Name of the logger to set.
        """
        self.logger_name = logger_name
        for value in self.__dict__.values():
            if isinstance(value, list):
                self._check_list(value, logger_name)
            else:
                attr = getattr(value, "set_logger_name_recursively", None)
                if callable(attr):
                    attr(logger_name)

    def get_logger(self) -> logging.Logger:
        """Gets the currently defined environment logger.

        If multiprocessing is part of the logger name the multiprocessing
        standard logger will be called.

        Returns:
            logging.Logger: logger which is currently to be used.
        """
        if self.logger_name and "multiprocessing" in self.logger_name:
            return multiprocessing.get_logger()
        return logging.getLogger(self.logger_name)

    # makes it that pydantic returns an error if unknown keywords are given
    class Config:
        """Used to add pydantic configurations."""

        #: Forbids superfluous keywords for object definitions.
        extra = pydantic.Extra.forbid
