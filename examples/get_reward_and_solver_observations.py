import logging
import pathlib
from typing import List, Any, Dict, Optional
from random import randint
from numpy.random import rand
from SbSOvRL.util.logger import VerbosityLevel, set_up_logger
from SbSOvRL.util.reward_helpers import load_json, write_json, spor_com_parse_arguments
from SbSOvRL.util.sbsovrl_types import ObservationType, RewardType, InfoType, DoneType
from pydantic.types import UUID4

validation_values:List[float] = [
    1.95,
    0.24,
    0.12,
    1.1,
    0.5
]

goal_range:float = 1.9
goal_offset:float = 0.1

acceptance_delta: float = 0.2


def read_csv():
    """Reads in output of the solver and calculates the needed ratio.
    
    Note: Directly taken from Michael Binder

    Returns:
        float: ratio between rights and left outflow
    """
    with open('outputIntegralRNG_2.csv') as f2:
        lines=f2.readlines()
        lines2=lines[2].split(',')
        fortnumb=lines2[2]
        pynumb=fortnumb.replace('D','E')
        pynumb2=float(pynumb)
    with open('outputIntegralRNG_3.csv') as f3:
        lines=f3.readlines()
        lines3=lines[2].split(',')
        fortnumb=lines3[2]
        pynumb=fortnumb.replace('D','E')
        pynumb3=float(pynumb)

    ratio23=pynumb2/pynumb3
    return float(ratio23)

def calculate_reward(ratio: float, goal: float, last_ratio: Optional[float] = None, **kwargs) -> float:
    """Function that actually calculates the reward.

    Note: Currently the reward is calculated the same way Michael calculated it.

    Args:
        ratio (float): Ratio the solver has computed.
        goal (float): Goal ratio.
        last_ratio (Optional[float], optional): To track progress the last ratio can also be input. Defaults to None.

    Returns:
        float: Calculated reward
    """
    info = {}
    if last_ratio is None:
        last_ratio = ratio
    reward = -0.2
    delta=abs(ratio-goal)
    delta_old=abs(last_ratio-goal)
    done = False
    
    if abs(abs(ratio) - abs(goal)) < acceptance_delta:
        reward = 5
        done = True
        info["reset_reason"] = "goal_achieved"
    else:
        if abs(delta) < abs(delta_old):
            reward = abs(ratio-last_ratio)/delta_old
        elif abs(delta) > abs(delta_old):
            reward = -0.5


    return reward, info, done


if __name__ == "__main__":
    args = spor_com_parse_arguments()

    pathlib.Path(f"{args.run_id}").mkdir(exist_ok=True, parents=True)
    local_variable_store_path = pathlib.Path(f"{args.run_id}/local_variable_store.json")
    
    reward_solver_logger = set_up_logger("reward_solver", pathlib.Path(str(args.run_id)), VerbosityLevel.INFO, console_logging=False)

    # initialize/reset variable store
    if args.reset or not local_variable_store_path.exists():
        local_variable_store = {}
        local_variable_store["last_ratio"] = None
        # TODO goal is only used here outside of the observation make goal reseting here
        if args.validation_run is not None:
            local_variable_store["goal"] = validation_values[args.validation_run]
            reward_solver_logger.debug(f"Setting predefined goal: {local_variable_store['goal']}.")
        else:
            # local_variable_store["goal"] = randint(2,20)/10
            local_variable_store["goal"] = (rand()*goal_range)+goal_offset
            reward_solver_logger.debug(f"Setting random Goal.")
        reward_solver_logger.info(f"Resetting with new goal: {local_variable_store['goal']}; The goal is set from the outside: {args.validation_run}")
        write_json(local_variable_store_path, local_variable_store)
    
    # load local variables
    local_variable_store = load_json(local_variable_store_path)

    # calculate the ratio
    ratio = read_csv()

    # calculate the reward
    reward, info, done = calculate_reward(ratio, **local_variable_store)

    # update step variable local_variables
    local_variable_store["last_ratio"] = ratio

    # define dict that is passed back
    return_dict = {
        "reward": reward,
        "done": done,
        "info": info,
        "observations": [
            ratio,
            local_variable_store["goal"] 
        ]
    }
    # {
    #     "reward": RewardType,
    #     "done": DoneType,
    #     "info": InfoType,
    #     "observations": ObservationType
    # }

    # store local variables
    write_json(local_variable_store_path, local_variable_store)
    
    logging.getLogger("reward_solver").info(f"The current intervals values are: {return_dict}")
    # write return values to console
    print(return_dict)