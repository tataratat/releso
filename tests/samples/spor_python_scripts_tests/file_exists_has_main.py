"""This file does not have a main function."""
from typing import Any, Dict, Tuple


def main(args, logger, func_data) -> Tuple[Dict[str, Any], Any]:
    """Dummy spor function."""
    return {
        "reward": 2,
        "done": False,
        "info": {},
        "observations": [1, 2, 3],
    }, func_data
