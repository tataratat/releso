import logging
import pathlib
from typing import Optional
from pydantic import BaseModel, Extra, Field
from pydantic.class_validators import validator
from typing import Any
from pydantic.fields import PrivateAttr
from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.util.logger import set_up_logger
# from SbSOvRL.verbosity import Verbosity
import datetime

class SbSOvRL_BaseModel(BaseModel):
    save_location: str    #: Definition of the save location of the logs and validation results
    logger_name: Optional[str] #: name of the logger. If this variable gives you trouble, the framework is at least a little bit buggy. Does not need to be set. And might be changed if set by user, if multi-environment training is utilized.
    
    # private fields
    _verbosity: Any = PrivateAttr(default=None) #: Defining the verbosity of the training process and environment loading
    

    def __init__(__pydantic_self__, **data: Any) -> None:
        if "verbosity_handover" in data:
            __pydantic_self__._verbosity = data["verbosity_handover"]
            for _, value in data.items():
                if type(value) is dict:
                    value["verbosity_handover"] = __pydantic_self__._verbosity
            data.pop("verbosity_handover")
        if "logger_name" in data:
            # if a logger name is present in the current object definition add this logger also to all object definition which are direct dependends
            for _, value in data.items():
                if type(value) is dict and "logger_name" not in value:
                    value["logger_name"] = data["logger_name"]    
        else:
            data["logger_name"] = "SbSOvRL_base_logger"
        if "save_location" in data:
            # if a logger name is present in the current object definition add this logger also to all object definition which are direct dependends
            for _, value in data.items():
                if type(value) is dict and "save_location" not in value:
                    value["save_location"] = data["save_location"]
        super().__init__(**data)

    @validator("save_location")
    @classmethod
    def convert_path_to_pathlib_and_add_datetime_if_applicable(cls, v):
        path = pathlib.Path(v.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
        
    # @validator("logger_name")
    # @classmethod
    # def register_logger_if_not_already_present(cls, v, values):
    #     if v is None:
    #         raise SbSOvRLParserException("BaseModel", "logger", "No logger name is given, this should never happen")
    #     else:
    #         if "save_location" in values:
    #             if not logging.getLogger(v).hasHandlers():
    #                 set_up_logger(v, values["save_location"])

    class Config:
        extra = Extra.forbid