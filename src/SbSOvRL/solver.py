from typing import Optional, List, Union
from pydantic.main import BaseModel
from pydantic.types import DirectoryPath, conint
from SbSOvRL.util.util_funcs import call_commandline

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

    def get_multiprocessor_command_prefix(self, core_count: int) -> str:
        """Returns the string that represents the call for multiprocessing. If multiprocessing is available.

        Args:
            core_count (int): Number of wanted cores.

        Returns:
            str: string representing the command line call or empty string if no multi processing is available.
        """
        return "" if self.multi_processor is None else self.multi_processor.get_command(core_count) + " "
    
    def start_solver(self, core_count: int) -> None:
        raise NotImplementedError("This function needst to be overloaded to work.")


class CommandLineSolver(Solver):
    execution_command: str
    command_options: List[str] = []
 
    def start_solver(self, core_count: int) -> float:
        command_str = self.get_multiprocessor_command_prefix(core_count) + self.get_command()
        command_str += self.execution_command + " " + " ".join(self.command_options)
        exit_code, output = call_commandline(command_str, self.working_directory)
        if exit_code != 1: # TODO should be also use the output somehow?
            pass
            # TODO generate return value
        return None

SolverTypeDefinition = Union[CommandLineSolver]