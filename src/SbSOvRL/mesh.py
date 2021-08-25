from pydantic import BaseModel, validator, FilePath
from typing import Literal, Dict, Any
# import wget
# from urllib.error import URLError
# import pathlib
from SbSOvRL.exceptions import SbSOvRLParserException
# from SbSOvRL.parser_functions import validate_file_path_to_absolute

class Mesh(BaseModel):
    location: Literal["local", "remote"] = "local"
    path: FilePath

    @validator("location")
    @classmethod
    def validate_location(cls, value: Literal["local", "remote"]) -> str:
        """Validation of the location. Currently only local is valid. Remote mights come in the future but is currently not planned.

        Args:
            value (str): [description]

        Raises:
            SbSOvRLParserException: [description]

        Returns:
            str: location string that is valid
        """
        if value is not "local":
            raise SbSOvRLParserException("Mesh", "location", "Currently the location must be local.")
        return value


    # @validator("path")
    # @classmethod
    # def validate_path_accessible(cls, value: str, values: Dict[str, Any], field: str) -> pathlib.Path:
    #     """Validates that the given value points to an existing file.

    #     Checks if path exists and that it is not a directory.

    #     Args:
    #         value (str): Value that is to be checked.
    #         values (Dict[str, Any]): All other already validates variables in a dict.
    #         field (str): Name of the field that is currently validated.

    #     Raises:
    #         SbSOvRLParserException: Error if the path is not valid.

    #     Returns:
    #         [pathlib.Path]: Absolute path to the mesh file.
    #     """
    #     ## downloads mesh file from remote server
    #     # if "location" in vars:
    #     #     if vars["location"] == "remote":
    #     #         try: # TODO check what exception gets thrown if download not possible (seems to be urllib exceptions)
    #     #             value = wget.download(value)
    #     #         except URLError as error:
    #     #             # file could not be downloaded # TODO either let URLError be raised or make custom exception for this
    #     #             pass
    #     # check if file exists
    #     return validate_file_path_to_absolute(value=value, parent="Mesh", field=field)
