"""Utility functions.

This files includes multiple functions and classes which are used in this
package but can not be directly associated to a single sub module.
"""
import datetime
import json
import os
import subprocess
from typing import Any, Optional

import numpy as np

from SbSOvRL.util.logger import logging
from SbSOvRL.util.sbsovrl_types import InfoType, ObservationType


class SbSOvRL_JSONEncoder(json.JSONEncoder):
    """Encodes numpy arrays.

    This encoder class is necessary to correctly encode numpy.ndarrays, bytes
    and numpy int values.
    """

    def default(self, o: Any) -> Any:
        """Default function.

        Args:
            o (Any): To be encoded.

        Returns:
            Any: Encoded.
        """
        if isinstance(o, np.ndarray):
            return o.astype(float).tolist()
        elif isinstance(o, np.int64):
            return int(0)
        elif isinstance(o, bytes):
            return o.decode("utf-8")
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError as err:
            print(type(o), o)
            return json.JSONEncoder.default(self, "")


def which(program: str) -> Optional[str]:
    """Finds if the given program is accessible or in the $PATH.

    Args:
        program (str): Program to look for

    Note:
        Verbatim taken from this stack overflow answer:
        https://stackoverflow.com/a/377028

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
    """Executes a command which is provided as a string in the command line.

    Author: Michael Binder (e1325632@student.tuwien.ac.at)
    """
    if logger is not None:
        logger.debug(f'Executing command {command} in {folder}')
    try:
        # try to execute the provided command in the shell
        output = subprocess.check_output(command, shell=True, cwd=folder)
        # print(output.decode('utf-8')) #utf-8 decoding macht öfters Probleme!
        exitcode = 0
    except subprocess.CalledProcessError as exc:
        # if anything went wrong, catch the error, report to the user and abort

        # print("error:",exc)
        # print("output:",exc.output)
        # print("returncode:",exc.returncode)
        # print("stderr:",exc.stderr)
        # print("stdout:",exc.stdout)
        output = exc.output
        if logger is not None:
            logger.error(f'Execution failed with return code {exc.returncode}')
        exitcode = exc.returncode
    return exitcode, output


def join_infos(old_info: InfoType, new_info: InfoType, logger_name: str):
    """Joins the provided old and new infos into a cohesive new dict.

    Args:
        old_info (InfoType): Already existing info dictionary which should be
        updated with the new infos.
        new_info (InfoType): Newly received info. Which needs to be added to
        the already existing infos.
        logger_name (str): LoggerName so that if necessary the log message is
        sent into the correct logging cue.
    """
    old_info.update(new_info)


def get_path_extension() -> str:
    """Creates a string which has the following characteristics.

    If not on a slurm system only the current time stamp is added
    If in a slurm job (non task array) the current time stamp + job id
    If in slurm task array job the job_id with subfolders for each task with
    the task_id_timestamp

    If on a slurm system the slurm information are read from the environment
    variables $SLURM_JOB_ID, $SLURM_ARRAY_JOB_ID, $SLURM_ARRAY_TASK_ID.

    Returns:
        str: See function documentation body for definition.
    """
    ret_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # check if slurm job is running
    if os.getenv("SLURM_CLUSTER_NAME"):
        # check if slurm task array is running
        if os.getenv("SLURM_ARRAY_TASK_COUNT") and \
            int(os.getenv("SLURM_ARRAY_TASK_COUNT")) > 1 and \
                os.getenv("SLURM_ARRAY_JOB_ID") and \
                os.getenv("SLURM_ARRAY_TASK_ID"):
            ret_str = os.getenv("SLURM_ARRAY_JOB_ID")+"/" + \
                os.getenv("SLURM_ARRAY_TASK_ID")+"_"+ret_str
        elif os.getenv("SLURM_JOB_ID"):  # default slurm job running
            ret_str += "_"+os.getenv("SLURM_JOB_ID")
    return ret_str


class ModuleImportRaiser():
    """Import error deferrer until it is actually called.

    Class used to have better import error handling in the case that a package
    package is not installed. This is necessary due to that some packages are
    not a dependency of `gustaf`, but some parts require them to function.
    Examples are `gustav` and `torchvision`.
    """

    def __init__(self, lib_name: str) -> None:
        """Constructor of object of class ModuleImportRaiser.

        Args:
            lib_name (str): Name of the library which can not be loaded. Will
            be inserted into the error message of the deferred import Error.
            Is not checked for correctness.
        """
        self._message = str(
            "Parts of the requested functionality in ReLeSO depend on the "
            f"external `{lib_name}` package which could not be found on "
            "your system. Please refer to the installation instructions "
            "for more information."
        )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Dummy method for object(args, kwargs).

        Is called when the object is called by object(). Will notify the user,
        that the functionality is not accessible and how to proceed to access
        the functionality.
        """
        raise ImportError(self._message)

    def __getattr__(self, __name: str) -> Any:
        """Dummy method for object.__name.

        Is called when any attribute of the object is accessed by object.attr.
        Will notify the user, that the functionality is not accessible and how
        to proceed to access the functionality.
        """
        if __name == "_ModuleImportRaiser__message":
            return object.__getattr__(self, __name[-8:])
        else:
            raise ImportError(self._message)

    def __setattr__(self, __name: str, __value: Any) -> None:
        """Dummy method for object.__name = __value.

        Is called when any attribute of the object is set by object.attr = new.
        Will notify the user, that the functionality is not accessible and how
        to proceed to access the functionality.
        """
        if __name == "_message":
            object.__setattr__(self, __name, __value)
        else:
            raise ImportError(self._message)

    def __getitem__(self, key):
        """Dummy method for object[key].

        Is called when the object is subscripted object[x]. Will notify the
        user, that the functionality is not accessible and how to proceed to
        access the functionality.
        """
        raise ImportError(self._message)
