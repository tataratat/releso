"""This file does not have a main function."""
from typing import Any, Dict, Tuple

from releso.util.reward_helpers import (
    load_json,
    spor_com_parse_arguments,
    write_json,
)


def main(args, logger=None, func_data=None) -> Tuple[Dict[str, Any], Any]:
    """Dummy spor function."""
    return {
        "reward": 2,
        "done": False,
        "info": {},
        "observations": [1, 2, 3],
    }, func_data


if __name__ == "__main__":
    arguments = spor_com_parse_arguments()
    save_path = (
        f"{arguments.base_save_location}/{arguments.environment_id}/"
        f"{arguments.run_id}.json"
    )

    func_data = load_json(save_path)

    return_data, func_data = main(arguments, None, func_data)

    write_json(save_path, func_data)

    print(return_data)
    exit(0)
