from typing import Union, Any, Dict
import json
from os import PathLike
import argparse
from pydantic.types import UUID4

def spor_com_parse_arguments():
    parser = argparse.ArgumentParser(description="Spline base Shape Optimization via Reinforcement Learning Toolbox. This is the basic script that can load a json file and run the resulting optimization problem.")
    parser.add_argument("-i", "--initialize", "--reset", dest="reset", action="store_true")
    parser.add_argument("-r", "--run", "--run_id", dest="run_id", action="store", type=UUID4, required=True)
    parser.add_argument("-v", "--validation_value", dest="validation_value", action="store", type=float, required=False)
    parser.add_argument("-j", "--additional_values", "--json_object", dest="json_object", action="store", type=str, required=False)
    args = parser.parse_args()
    if args.json_object:
        args.json_object = spor_com_additional_informations(args.json_object)
    return args

def spor_com_additional_informations(j_str: str) -> Dict[str, Any]:
    a = json.loads(j_str)
    return a

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
