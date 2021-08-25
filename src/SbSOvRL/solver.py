from typing import Optional, List, Union
from pydantic.main import BaseModel
from pydantic.types import DirectoryPath, conint
import subprocess
import os

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
    # TODO how is the return value handled. There is not a direct return value from the function but a file needs to be opened and than a specific value read. And some other processing.

    def _call_commandline(command,folder):
        """
            Executes a command which is provided as a string in the command line
            Author: Michael Binder (e1325632@student.tuwien.ac.at)
        """

        print('-- Executing command {} in {}'.format(command, os.getcwd()))
        try:
            # try to execute the provided command in the shell
            output = subprocess.check_output(command, shell=True,cwd=folder)
            #print(output.decode('utf-8')) #utf-8 decoding macht öfters Probleme!
            exitcode=0
        except subprocess.CalledProcessError as exc:
            # if anything went wrong, catch the error, report to the user and abort
            print('-- Execution failed with return code {}'.format(exc.returncode)) # TODO better logging of errors need to be implemented
            exitcode=exc.returncode
        return exitcode, output


    def start_solver(self, core_count: int) -> float:
        command_str = self.get_multiprocessor_command_prefix(core_count) + self.get_command()
        command_str += self.execution_command + " " + " ".join(self.command_options)
        print(command_str)
        exit_code, output = self._call_commandline(command_str, self.working_directory)
        if exit_code != 1: # TODO should be also use the output somehow?
            pass
            # TODO generate return value
        return None

SolverTypeDefinition = Union[CommandLineSolver]