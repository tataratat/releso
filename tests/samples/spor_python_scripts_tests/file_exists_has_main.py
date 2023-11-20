"""This file does not have a main function."""
from typing import Any, Dict, Tuple


def main(args, logger, func_data) -> Tuple[Dict[str, Any], Any]:
    """Dummy spor function."""
    ret_dict = {
        "reward": 2,
        "done": False,
        "info": {},
        "observations": [1, 2, 3],
    }
    if func_data is None or args.reset:
        func_data = 0
    else:
        func_data += 1
    if func_data > 10:
        ret_dict["done"] = True
        ret_dict["info"]["reset_reason"] = "func_data > 10"
    return ret_dict, func_data
