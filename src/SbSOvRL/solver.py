from typing import Optional, List, Union, Any, Dict, Tuple
from pydantic.fields import PrivateAttr
from pydantic.main import BaseModel
from pydantic.types import DirectoryPath, conint
from SbSOvRL.util.util_funcs import call_commandline
from SbSOvRL.reward_parser import RewardFunctionTypes
from SbSOvRL.util.logger import set_up_logger

parser_logger = set_up_logger("SbSOvRL_parser")
environment_logger = set_up_logger("SbSOvRL_environment")


class MultiProcessor(BaseModel):
    command: str = "mpiexec -np"
    max_core_count: conint(ge=1)

    def get_command(self, core_count: int) -> str:
        """Returns the string that represents the call for multiprocessing.

        Args:
            core_count (int): Number of wanted cores.

        Returns:
            str: string representing the command line call
        """
        return self.command + " " + max(1, min(core_count, self.max_core_count))


class Solver(BaseModel):
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

    def start_solver(self, core_count: int = 0, reset: bool = False) -> Tuple[Dict[str, Any], bool]:
        done = False
        parser_logger.error(f"The working directory is: {self.working_directory}")
        command_str = self.get_multiprocessor_command_prefix(
            core_count)

        command_str += self.execution_command + \
            " " + " ".join(self.command_options)
        
        self._exit_code, self._output = call_commandline(
            command_str, self.working_directory, environment_logger)
        if self._exit_code != 0:  # TODO should be also use the output somehow?
            environment_logger.info("Solver thrown error, episode will end now...")
            reward_observation_obj = {
                "reward": -5,
                "observation": []
            }
            done = True
        else:
            # return value of reward function
            additional_parameter = []
            if reset:
                additional_parameter.append("-i")
            reward_observation_obj = self.get_reward_solver_observations(
                additional_parameter)
        return reward_observation_obj, done


SolverTypeDefinition = Union[CommandLineSolver]
