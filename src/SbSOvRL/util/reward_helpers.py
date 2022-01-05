from typing import Union, Any, Dict
import json
from os import PathLike

def load_json(f_n: Union[PathLike, str]) -> Dict[str, Any]:
    """Loads data from a given file as a json object.

    Args:
        f_n (Union[PathLike, str]): Path to the file where the json encoded data is stored.

    Returns:
        Dict[str, Any]: Data loaded from the given file.
    """
    with open(f_n, "r") as rf:
        return json.load(rf)

def write_json(f_n: Union[PathLike, str], obj_dict: Dict[str, Any]):
    """Write the obj_dict to the given file (json encoded). Can be used to store persistent variables between calls to the same script. 

    Args:
        f_n (Union[PathLike, str]): Path to the file where the json encoded data is to be stored.
        obj_dict (Dict[str, Any]): Data that is to be stored
    """
    with open(f_n, "w") as wf:
        json.dump(obj_dict, wf)
