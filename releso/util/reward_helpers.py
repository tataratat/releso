"""SPOR object script helper functions.

This files holds functions which can be used in custom command line spor object
scripts.
Tried to keep this as package independent as possible, to make it easier to
copy these functions into your own scripts. If you need to have other python
versions/special environment to run your scripts in. The pydantic dependency
can be exchanged for a string.
"""

import argparse
import json
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic.types import UUID4


def spor_com_parse_arguments(
    own_arguments: Optional[List[str]] = None,
) -> argparse.Namespace:
    """Parses the spor com arguments.

    The result is returned in a namespace object for easy access.

    Returns:
        argparse.Namespace: [description]
    """
    parser = argparse.ArgumentParser(
        description="Reinforcement Learning based Shape Optimization"
        " Toolbox. This is the basic script that can load a json file and run"
        " the resulting optimization problem."
    )
    parser.add_argument(
        "-i",
        "--initialize",
        "--reset",
        dest="reset",
        action="store_true",
        help="Should the script perform the step necessary for a reset.",
    )
    parser.add_argument(
        "-r",
        "--run",
        "--run_id",
        dest="run_id",
        action="store",
        type=UUID4,  # can also be a string. If pydantic is not available.
        required=True,
        help="Id of the SPOR_Communication interface for this "
        "specific spor object.",
    )
    parser.add_argument(
        "-v",
        "--validation_value",
        dest="validation_value",
        action="store",
        type=float,
        required=False,
        help="If during validation the current validation value is passed to "
        "the script by the variable. If not present not currently in "
        "validation.",
    )
    parser.add_argument(
        "-j",
        "--additional_values",
        "--json_object",
        dest="json_object",
        action="store",
        type=str,
        required=False,
        help="Currently available step information including observations, "
        "done, reward, info. Will automatically be parsed into a dict. Only "
        "present if spor step is configured to sent it.",
    )
    parser.add_argument(
        "-l",
        "--base_save_location",
        dest="base_save_location",
        action="store",
        type=str,
        required=True,
        help="Path pointing to the save location of all permanent records of "
        "this trainings run. Please save the logs here.",
    )
    parser.add_argument(
        "-e",
        "--environment_id",
        dest="environment_id",
        action="store",
        type=str,
        required=True,
        help="ID of the environment this call is "
        "coming from. NOT unique to this spor object.",
    )
    if own_arguments is not None:
        args = parser.parse_args(own_arguments)
    else:
        args = parser.parse_args()  # pragma: no cover
    if args.json_object:
        args.json_object = spor_com_additional_information(args.json_object)
    return args


def spor_com_additional_information(j_str: str) -> Dict[str, Any]:
    """Converts the given string (should hold a json string) into a dictionary.

    This function is currently only used for the spor_com_parse_arguments
    function.

    Args:
        j_str (str): string holding a json definition

    Returns:
        Dict[str, Any]: Dict created from the json definition
    """
    a = json.loads(j_str[1:-1])
    return a


def load_json(f_n: Union[PathLike, str]) -> Dict[str, Any]:
    """Loads data from a given file as a json object.

    Will create a dummy file if it does not exist.

    Args:
        f_n (Union[PathLike, str]): Path to the file where the json encoded
        data is stored.

    Returns:
        Dict[str, Any]: Data loaded from the given file.
    """
    path = Path(f_n)
    if not (path.exists() and path.is_file()):
        empty_dict: Dict[str, Any] = {}
        try:
            with open(path, "w") as wf:
                json.dump(empty_dict, wf)
        except FileNotFoundError as err:
            raise RuntimeError(
                "You have to create the folder for the json first."
            ) from err
    with open(f_n) as rf:
        return json.load(rf)


def write_json(f_n: Union[PathLike, str], obj_dict: Dict[str, Any]):
    """Write the obj_dict to the given file (json encoded).

    Can be used to store persistent variables between calls to the same script.

    Args:
        f_n (Union[PathLike, str]): Path to the file where the json encoded
        data is to be stored.
        obj_dict (Dict[str, Any]): Data that is to be stored.
    """
    with open(f_n, "w") as wf:
        json.dump(obj_dict, wf)
