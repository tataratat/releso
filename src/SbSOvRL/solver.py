import os
from typing import Literal, Optional, List, Union, Any, Dict, Tuple
from pydantic.fields import PrivateAttr
from pydantic.types import DirectoryPath, conint
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

        return f"{self.command} {str(max(1, min(core_count, self.max_core_count)))}"

class MPIClusterMultiProcessor(MultiProcessor):
    location: Literal["cluster"]
    command: str = "$MPIEXEC"
    mpi_flags_variable: str = "$FLAGS_MPI_BATCH"

    def get_command(self, core_count: int) -> str:
        # logging.getLogger("SbSOvRL_environment").warning("Using Cluster mpi")
        # check if environment variable for mpi has the correct core number.
        environ_mpi_flags: List = str(os.environ[self.mpi_flags_variable.replace("$", '')]).split()
        if len(environ_mpi_flags) != 2:
            logging.getLogger("SbSOvRL_environment").error("The environment variable for mpi cluster did not look as expected. Please use standard MultiProcessor or revise the code.")
            raise RuntimeError("Could not complete Task. Please see log for more informations.")
        core_qualifier, set_core_count = environ_mpi_flags
        if set_core_count != core_count:
            os.environ[self.mpi_flags_variable.replace("$", '')] = f"{core_qualifier} {max(1, min(core_count, self.max_core_count))}"
        logging.getLogger("SbSOvRL_environment").warning(f"Using Cluster mpi with command {os.environ[self.command.replace('$', '')]} and additional flags {os.environ[self.mpi_flags_variable.replace('$', '')]}")
        return f"{self.command} {self.mpi_flags_variable}"

class Solver(SbSOvRL_BaseModel):
    multi_processor: Optional[Union[MultiProcessor, MPIClusterMultiProcessor]] = None
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

    def start_solver(self, core_count: int = 0, reset: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        raise NotImplementedError(
            "This function needst to be overloaded to work.")

    def get_reward_solver_observations(self, additional_parameter: List[str] = []) -> Dict[str, Any]:
        return self.reward_output.get_reward(additional_parameter=additional_parameter)


class CommandLineSolver(Solver):
    execution_command: str
    command_options: List[str] = []

    def start_solver(self, core_count: int = 0, reset: bool = False, new_goal_value: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
        # setup uuid the first time this function gets called.
        
        done = False
        info = {}
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
            info["reset_reason"] = "solver_error"
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
            info.update(reward_observation_obj["info"])
        return reward_observation_obj, info, done


SolverTypeDefinition = Union[CommandLineSolver]
