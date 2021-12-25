"""
Solver Postprocessing Observation Reward (SPOR) is represents all steps which follow after completing the Free Form Deformation (FFD), where the base mesh is deformed with a spline.

This part of the SbSOvRL environment definition is a list of steps the environment will run through in series. Each step can be configured to use the SbSOvRL communication protocol.

A more in depth documentation of the SPOR concept is given here :ref:`SPOR Communication Interface <sporcominterface>`
"""

from pydantic.fields import PrivateAttr
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.util.logger import logging
from SbSOvRL.util.sbsovrl_types import InfoType, ObservationType, RewardType, StepReturnType
from SbSOvRL.exceptions import SbSOvRLParserException
import os, pathlib
import numpy as np
from typing import List, Literal, Optional, Any
from pydantic import conint, validator

class MultiProcessor(SbSOvRL_BaseModel):
    """
    The MultiProcessor gives access to command prefixes which can enable parallelization. Currently only tested for MPI. 
    
    Note: 
        The cluster uses PATH variables to correctly call mpi. IF you want to use MPI multiprocessing on the cluster please use :py:class:`SbSOvRL.spor.MPIClusterMultiProcessor`.
    """
    command: str = "mpiexec -np"  #: command prefix which is needed to parallelize the give task
    max_core_count: conint(ge=1)  #: Max core count to use for this parallelization option. This variable is not shared between tasks or environments so the user has to split these up before hand.

    def get_command(self, core_count: int) -> str:
        """Returns the string that represents the call for multiprocessing.

        Args:
            core_count (int): Number of wanted cores.

        Returns:
            str: string representing the command line call prefix
        """

        return f"{self.command} {str(max(1, min(core_count, self.max_core_count)))}"

class MPIClusterMultiProcessor(MultiProcessor):
    """
        Muti-processor extension for the cluster MPI call. 
    """
    location: Literal["cluster"]  #: The location title needs to be set to ``cluster`` so that 
    command: str = "$MPIEXEC" #: command prefix which is needed to parallelize the give task
    mpi_flags_variable: str = "$FLAGS_MPI_BATCH"    #: additional flags for the mpi call. Can also be a PATH variable.

    def get_command(self, core_count: int) -> str:
        """Returns the string that represents the call for multiprocessing.
            
        Note:
            Since the cluster uses PATH variables for its computation. This function will read these in and use them correspondingly. But will not change them. This limitation is done so that there is less interference during Multi-Environment training.

        Args:
            core_count (int): Number of wanted cores.

        Raises:
            RuntimeError: This error is thrown if the mpi_flags PATH variable is not of the shape excpected by the function. Expected are 2 parts -np [core_count] where -np can be anything and the core_count should be a number.

        Returns:
            str: string representing the command line call prefix
        """
        # logging.getLogger("SbSOvRL_environment").warning("Using Cluster mpi")
        # check if environment variable for mpi has the correct core number.
        environ_mpi_flags: List = str(os.environ[self.mpi_flags_variable.replace("$", '')]).split()
        local_mpi_flags = self.mpi_flags_variable
        if len(environ_mpi_flags) != 2:
            logging.getLogger("SbSOvRL_environment").error("The environment variable for mpi cluster did not look as expected. Please use standard MultiProcessor or revise the code.")
            raise RuntimeError("Could not complete Task. Please see log for more information.")
        core_qualifier, set_core_count = environ_mpi_flags
        if set_core_count != core_count:
            local_mpi_flags = f"{core_qualifier} {max(1, min(core_count, self.max_core_count))}"
        logging.getLogger("SbSOvRL_environment").info(f"Using Cluster mpi with command {os.environ[self.command.replace('$', '')]} and additional flags {local_mpi_flags}")
        return f"{self.command} {local_mpi_flags}"

class SPORObject(SbSOvRL_BaseModel):
    """
        Base class for all possible SPOR object classes. Theses objects represent the possible steps that can be/are executed after the FFD finished.

        The following arguments need to be set for SPOR objects, further parameters are use-case specific.
    """
    stop_after_error: bool = True #: Whether to stop after this step if this step has thrown an error. This is important if 
    reward_on_error: float  #: Reward which is returned if any kind of error is thrown during completing the defined task.
    reward_on_completion: Optional[float] = None  #: Reward which is returned if the task completed with out an error. If this is set this reward will overwrite any other reward this function will generate. (Implementation specific might differ for child classes see documentation). If *None* no default reward is returned when task is finished. Defaults to *None*.

    def run(self, validation_id: Optional[int] = None) -> StepReturnType:
        """This function is called to complete the defined step of the current SPORObject. This functions needs to be overloaded by all child classes.

        Note:
            Please use these default values for the return values: observation = None, reward = 0, done = False, info = dict()

        Args:
            validation_id (Optional[int], optional): During validation the validation id signifies the validation episode. If none the current episode is not a validation episode. Defaults to None.

        Returns:
            StepReturnType: The full compliment of the step information are returned. The include observation, reward, done, info. 
        """
        return NotImplementedError

    def reset(self, validation_id: Optional[int] = None) -> StepReturnType:
        """This function is called if the current episode ends and needs to be reset.

        Args:
            validation_id (Optional[int], optional): [description]. Defaults to None.

        Returns:
            StepReturnType: [description]
        """
        return NotImplementedError

class SPORObjectCommandLine(SPORObject):
    """
        Base class for all possible SPOR object classes which use the command line. This class is meant to add some default functionality, so that users do not need to add them themselfs. These are:

        1. access to the (MPI) mulit-processors via :py:class:`SbSOvRL.spor.MultiProcessor`
        2. command line command
        3. additional flags
        4. working directory
        5. whether or not to use the :ref:`SPOR Communication Interface <sporcominterface>`

    """
    multi_processor: Optional[MultiProcessor] = None   #: Definition of the multi-processor. Defaults to *None*.
    use_communication_interface: bool = False   #: whether or not to use the SPOR communication interface. Defaults to *False*.
    working_directory: str  #: Path to the directory in which the program should run in helpful for programs using relative paths. When {} appear in the string a UUID will be inserted for multiprocessing purposes.
    execution_command: str  #: Command which calls the program/script for this SPOR step.  
    command_options: List[str] = [] #: Command line options to add to the execution command. These do not include those from the communication interface these will be, if applicable, added separately.


    @validator("working_directory")
    def validate_working_directory_path(cls, v):
        # check if placeholder for multiprocessing is available if not use v to validate directory path
        path = pathlib.Path(v.split("{]")[0]) if "{}" in v else v
        if not (path.is_dir() and path.exists()):
            raise SbSOvRLParserException("SPORObjectCommandline", "unknown", f"The work_directory path {v} is a valid directory. When using multiprocessing placeholder please make sure that the directory before the placeholder is valid.")
        return v

    def run(self, validation_id: Optional[int] = None) -> StepReturnType:
        pass

           
class SPORList(SbSOvRL_BaseModel):
    """
    The SPORList defines the steps which need to be taken for after the FFD is done. These can include the fluid solver, post-processing steps, reward generation, additional logging, etc. 

    If a step sends a stop signal all consecutive steps will be ignored and the 
    Each step can generate observations, rewards and info for the current episode step.  These are aggregated as follows.
    """
    steps: List[SPORObject]  #: (List[SPORObject]) List of SPOR objects. Each SPOR object is a step in the environment.
    reward_aggregation: Literal["sum", "min", "max", "mean", "minmax"]   #: Definition on how the reward should be aggregated. Definition for each option is given :doc:`here <spor>`.

    _rewards: List[RewardType] = PrivateAttr(default_factory=list())
    _observations: Any = PrivateAttr(default=None)
    _info: InfoType = PrivateAttr(default_factory=dict())

    def _compute_reward(self) -> RewardType:
        """Aggregates and computes the reward of the SPORList. The aggregation method is defined in the variable :py:obj:`SbSOvRL.spor.SPORList.reward_aggregation`.

        Raises:
            RuntimeError: If aggregation method is unknown an error is raised.

        Returns:
            RewardType: Total reward of the SPORList
        """
        if len(self._rewards) > 0:
            if self.reward_aggregation == "sum":
                return np.sum(self._rewards)
            elif self.reward_aggregation == "mean":
                return np.mean(self._rewards)
            elif self.reward_aggregation == "min":
                return np.min(self._rewards)
            elif self.reward_aggregation == "minmax":
                return self._rewards[np.argmax(np.abs(self._rewards))]
            elif self.reward_aggregation == "max":
                return np.max(self._rewards)
            else:
                err_mesg = f"The given reward aggregation method - {self.reward_aggregation} - is not supported."
                logging.getLogger(self._logger_name).error(err_mesg)
                raise RuntimeError(err_mesg)
        else:
            logging.getLogger(self._logger_name).warning("No rewards given. Setting reward to zero (0). Please check if reward should be given but did not register.")
            return 0

    def _add_observation(self, observation: ObservationType):
        """Adds the new observations to the already received observations. Currently only :py:`numpy.ndarrays` are permissable.

        Args:
            observation (ObservationType): Observations which should be added to the internal observation field.
        """
        if self._observations:
            if type(observation) is np.ndarray and type(self._observations) is np.ndarray:
                obs = [self._observations, observation]
                self._observations = np.ndarray(obs)
            else:
                print(type(observation))
                logging.getLogger(self.logger_name).warning(f"Conversion from {type(observation)} to {type(self._observations)} has currently no handler to stack observations. Please add one.")
        else:
            self._observations = observation

    def _add_info(self, info: InfoType):
        """Updates the internal Info field with the given info.

        Args:
            info (InfoType): Newly received info. Which needs to be added to the internal info field.
        """
        self._info.update(info)

    def run(self, validation_id: Optional[int] = None) -> StepReturnType:
        """Runs through all steps of the list and handles each output. When done all received observations and infos are jointly returned with the reward and the indicator whether or not the episode should be terminated.

        Args:
            validation_id (Optional[int], optional): If currently in validation the validation needs to be supplied for steps which need this. If none given no validation is currently performed. Defaults to None.

        Returns:
            StepReturnType: Results of all steps.
        """
        self._rewards = list()
        done = False
        for step in self.steps:
            observation, reward, done, info = step.run(validation_id)
            if observation is not None:
                self._add_observation(observation)
            if reward is not None:
                self._rewards.append(reward)
            if info:
                self._add_info(info)
            if done:
                break
        return self._observations, self._compute_reward(), done, self._info
