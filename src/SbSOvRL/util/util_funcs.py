"""
This files includes multiple functions which are used in this package but can not be directly associated to a single sub module.
"""
from typing import Optional
import subprocess
import os
import numpy as np
from SbSOvRL.util.logger import logging

from SbSOvRL.util.sbsovrl_types import InfoType, ObservationType


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

def join_observations(old_observations: ObservationType, new_observations: ObservationType, logger_name: str):
    """Adds the new observations to the already received observations. Currently only :py:class:`numpy.ndarrays` are permissable.

    Join methods:
        If :py:class:`numpy.ndarrays` and :py:class:`numpy.ndarrays` are joined the numpy function :py:meth:`numpy.append` is used. This will flatten all arrays and append the new array to the end of the old array. The resulting shape will be (x,) where x = prod(new_obs.shape)*prod(old_obs.shape)

    Args:
        old_observations (ObservationType): Already existing observations on which the new observations should be added to.
        new_observations (ObservationType): Observations which should be added to the old_observation field.
    """
    if old_observations:
        if type(new_observations) is np.ndarray and type(old_observations) is np.ndarray:
            old_observations = np.append(old_observations, new_observations)
        else:
            # print(type(new_observations))
            logging.getLogger(logger_name).warning(f"Conversion from {type(new_observations)} to {type(old_observations)} has currently no handler to stack observations. Please add one.")
    else:
        old_observations = new_observations

def join_infos(old_info: InfoType, new_info: InfoType, logger_name: str):
    """Updates the old Info field with the new info.

    Args:
        old_info (InfoType): Already existing info dictionary which should be updated with the new infos.
        new_info (InfoType): Newly received info. Which needs to be added to the already existing infos.
    """
    old_info.update(new_info)