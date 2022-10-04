import logging
import pathlib
from typing import List, Any, Dict, Optional, Tuple
# from random import randint
# from numpy.random import rand
import numpy as np
from SbSOvRL.util.logger import VerbosityLevel, set_up_logger
from SbSOvRL.util.reward_helpers import load_json, write_json, spor_com_parse_arguments


def main(args, reward_solver_log) -> Dict[str, Any]:
    pathlib.Path(f"{args.run_id}").mkdir(exist_ok=True, parents=True)

    # do you need persistent local storage?
    local_variable_store_path = pathlib.Path(f"{args.run_id}/local_variable_store.json")

    # initialize/reset variable store
    if args.reset or not local_variable_store_path.exists():
        local_variable_store = {}
        # initialize persistent local variable store


        write_json(local_variable_store_path, local_variable_store)
    
    # load local variables
    local_variable_store = load_json(local_variable_store_path)

    # run your code here

    reward, done, info, observations = (0, False, {}, [])

    # define dict that is passed back
    return_dict = {
        "reward": reward,
        "done": done,
        "info": info,
        "observations": observations
    }

    # store persistent local variables
    write_json(local_variable_store_path, local_variable_store)
    
    logging.getLogger("reward_solver").info(f"The current intervals values are: {return_dict}")
    # write return values to console
    return return_dict


if __name__ == "__main__":
    args = spor_com_parse_arguments()
    
    
    base_path = pathlib.Path(str(args.base_save_location))/str(args.environment_id)/str(args.run_id)
    base_path.mkdir(exist_ok=True, parents=True)
    local_variable_store_path = base_path / "local_variable_store.json"
    
    reward_solver_logger = set_up_logger("reward_solver", base_path, VerbosityLevel.INFO, console_logging=False)

    return_dict = main(args, reward_solver_logger)
    
    print(return_dict)