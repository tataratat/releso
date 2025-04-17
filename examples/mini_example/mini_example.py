from collections import namedtuple
from releso.util.reward_helpers import spor_com_parse_arguments, write_json, load_json
import numpy as np
from logging import Logger
from typing import Optional, Any
from pathlib import Path
import os

def main(args: namedtuple, logger: Logger, func_data: Optional[Any]):
    done = False
    info = {}

    # if add_step_information is not true, the json_object is None
    # but it is needed to calculate the reward
    if not args.json_object:
        print("No additional payload, please provide the needed payload.")

    # setup the func_data object, it is not used in this example
    if func_data is None:
        func_data = dict()

    # calculate the reward
    reward = sum(np.array(args.json_object["info"]["geometry_information"]))[0]

    # if reward is very close to the maximum of 15 it is considered as done
    if reward >= (15-1e-7):
        logger.warning(f"This is triggered why? : {reward}")
        reward = 30
        done = True
        info["reset_reason"] = "goal_reached"

    logger.info(
        f"{args.json_object['info']['geometry_information']}, Sum: {sum(np.array(args.json_object['info']['geometry_information']))[0]}, Reward: {reward}"
    )

    return {
            "reward": reward,
            "done": done,
            "info": info,
            "observations": []
        }, func_data

# Add option of running the script manually, or with original command line
# spor step in ReLeSO.
if __name__ == "__main__":
    args = spor_com_parse_arguments()
    if not args.json_object:
        print("No additional payload, please provide the needed payload.")

    #
    path = Path(f"{os.getcwd()}/{args.run_id}")
    path.mkdir(exist_ok=True, parents=True)

    local_variable_store_path = path/"local_variable_store.json"
    if not path.exists():
        func_data = {
            "last_error": 0
        }
        write_json(local_variable_store_path, func_data)

    func_data = load_json(local_variable_store_path)

    step_data, func_data = main(args, False, func_data)

    write_json(path, func_data)

    print(step_data)
