from typing import Optional, List, Union, Any, Dict, Tuple
import uuid
from pydantic.fields import PrivateAttr
from pydantic.types import UUID4, DirectoryPath, conint
from SbSOvRL.util.util_funcs import call_commandline
from SbSOvRL.reward_parser import RewardFunctionTypes
import logging
from SbSOvRL.base_model import SbSOvRL_BaseModel


class MultiProcessor(SbSOvRL_BaseModel):
    command: str = "mpiexec -np"
    max_core_count: conint(ge=1)

    def get_command(self, core_count: int) -> str:
        """Returns the string that represents the call for multiprocessing.

        Args:
            core_count (int): Number of wanted cores.

        Returns:
            str: string representing the command line call
        """
        return self.command + " " + str(max(1, min(core_count, self.max_core_count)))


class Solver(SbSOvRL_BaseModel):
    multi_processor: Optional[MultiProcessor] = None
    working_directory: DirectoryPath
    reward_output: RewardFunctionTypes

    _output: Any = PrivateAttr(default=None)
    _exit_code: Any = PrivateAttr(default=None)

    def get_multiprocessor_command_prefix(self, core_count: int) -> str:
        """Returns the string that represents the call for multiprocessing. If multiprocessing is available.

        Args:
            core_count (int): Number of wanted cores.

        Returns:
            str: string representing the command line call or empty string if no multi processing is available.
        """
        return "" if (self.multi_processor is None or core_count == 0) else (self.multi_processor.get_command(core_count) + " ")

    def start_solver(self, core_count: int = 0, reset: bool = False) -> Tuple[Dict[str, Any], bool]:
        raise NotImplementedError(
            "This function needst to be overloaded to work.")

    def get_reward_solver_observations(self, additional_parameter: List[str] = []) -> Dict[str, Any]:
        return self.reward_output.get_reward(additional_parameter=additional_parameter)


class CommandLineSolver(Solver):
    execution_command: str
    command_options: List[str] = []

    def start_solver(self, core_count: int = 0, reset: bool = False, new_goal_value: Optional[float] = None) -> Tuple[Dict[str, Any], bool]:
        # setup uuid the first time this function gets called.
        
        done = False
        command_str = self.get_multiprocessor_command_prefix(
            core_count)
        command_str += self.execution_command + \
            " " + " ".join(self.command_options)
        
        self._exit_code, self._output = call_commandline(
            command_str, self.working_directory, logging.getLogger("SbSOvRL_environment"))
        if self._exit_code != 0:  # TODO should be also use the output somehow?
            logging.getLogger("SbSOvRL_environment").info("Solver thrown error, episode will end now...")
            reward_observation_obj = {
                "reward": -5,
                "observation": []
            }
            done = True
        else:
            additional_parameter = []
            # return value of reward function
            if reset:
                additional_parameter.append("-i")
                if new_goal_value:
                    additional_parameter.append("-g")
                    additional_parameter.append(str(new_goal_value))
            reward_observation_obj = self.get_reward_solver_observations(
                additional_parameter)
            done = done or reward_observation_obj["done"]
        return reward_observation_obj, done


SolverTypeDefinition = Union[CommandLineSolver]
