from typing import Optional
import subprocess
import os


def which(program: str) -> Optional[str]:
    """Finds if the given program is accessible or in the $PATH

    Args:
        program (str): Program to look for

    Note: 
        Verbatim taken from this stackoverflow answer: https://stackoverflow.com/a/377028

    Returns:
        Optional[str]: None if not found else execution file.
    """
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

    
def call_commandline(command, folder, logger=None):
    """
        Executes a command which is provided as a string in the command line
        Author: Michael Binder (e1325632@student.tuwien.ac.at)
    """

    if logger is not None:
        logger.debug(f'Executing command {command} in {folder}')
    try:
        # try to execute the provided command in the shell
        output = subprocess.check_output(command, shell=True,cwd=folder)
        #print(output.decode('utf-8')) #utf-8 decoding macht öfters Probleme!
        exitcode=0
    except subprocess.CalledProcessError as exc:
        # if anything went wrong, catch the error, report to the user and abort
        output = None
        if logger is not None:
            logger.error(f'Execution failed with return code {exc.returncode}')
        exitcode=exc.returncode
    return exitcode, output