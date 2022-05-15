"""
Solver Postprocessing Observation Reward (SPOR) is represents all steps which follow after completing the Free Form Deformation (FFD), where the base mesh is deformed with a spline.

This part of the SbSOvRL environment definition is a list of steps the environment will run through in series. Each step can be configured to use the SbSOvRL communication protocol.

A more in depth documentation of the SPOR concept is given here :ref:`SPOR Communication Interface <sporcominterface>`
"""
from SbSOvRL.util.reward_helpers import spor_com_parse_arguments
import importlib
from SbSOvRL.util.logger import VerbosityLevel, set_up_logger
import json
import shutil
from pydantic.fields import PrivateAttr
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.util.sbsovrl_types import ObservationType, RewardType, StepReturnType
from SbSOvRL.exceptions import SbSOvRLParserException
import os, pathlib
import numpy as np
from typing import List, Literal, Optional, Any, Dict, Union, Tuple
from gym import Space
from gym.spaces import Box
from pydantic import NoneBytes, conint, validator, UUID4
from uuid import uuid4
import sys
from ast import literal_eval
from SbSOvRL.util.util_funcs import call_commandline, join_infos, SbSOvRL_JSONEncoder
from timeit import default_timer as timer

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

        return f"{self.command} {str(max(1, min(core_count, self.max_core_count)))}" if core_count > 1 else ""

class MPIClusterMultiProcessor(MultiProcessor):
    """
        Multi-processor extension for the cluster MPI call. 
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
        # self.get_logger().warning("Using Cluster mpi")
        # check if environment variable for mpi has the correct core number.
        if core_count == 1:
            return ""
        environ_mpi_flags: List = str(os.environ[self.mpi_flags_variable.replace("$", '')]).split()
        local_mpi_flags = self.mpi_flags_variable
        if len(environ_mpi_flags) != 2:
            self.get_logger().error("The environment variable for mpi cluster did not look as expected. Please use standard MultiProcessor or revise the code.")
            raise RuntimeError("Could not complete Task. Please see log for more information.")
        core_qualifier, set_core_count = environ_mpi_flags
        if set_core_count != core_count:
            local_mpi_flags = f"{core_qualifier} {max(1, min(core_count, self.max_core_count))}"
        self.get_logger().debug(f"Using Cluster mpi with command {os.environ[self.command.replace('$', '')]} and additional flags {local_mpi_flags}")
        return f"{self.command} {local_mpi_flags}"

class ObservationDefinition(SbSOvRL_BaseModel):
    name: str
    value_min: float
    value_max: float

    def get_observation_definition(self) -> Tuple[str, Space]:
        return (self.name, Box(self.value_min, self.value_max, shape=(1), dtype=np.float32))
    
    def get_default_observation(self) -> np.ndarray:
        return np.ones((1,))*self.value_min

class ObservationDefinitionMulti(ObservationDefinition):
    observation_shape: List[int]
    value_type: Literal["float", "CNN"]

    def get_observation_definition(self) -> Tuple[str, Space]:
        if self.value_type == "CNN":
            return (self.name, Box(low=0, high=255, shape=self.observation_shape, dtype=np.uint8))
        return (self.name, Box(self.value_min, self.value_max, shape=(self.observation_shape), dtype=np.float32))
    
    def get_default_observation(self) -> np.ndarray:
        if self.value_type == "CNN":
            return np.zeros(self.observation_shape, dtype=np.uint8)
        return np.ones(self.observation_shape, dtype=np.float32)*self.value_min


class SPORObject(SbSOvRL_BaseModel):
    """
        Base class for all possible SPOR object classes. Theses objects represent the possible steps that can be/are executed after the FFD finished.

        The following arguments need to be set for SPOR objects, further parameters are use-case specific.
    """
    name: str   #: name of the SPOR step. Makes so that it is possible to distinguish different steps more easily.
    stop_after_error: bool = True #: Whether to stop after this step if this step has thrown an error. This is important if 
    reward_on_error: float  #: Reward which is returned if any kind of error is thrown during completing the defined task.
    reward_on_completion: Optional[float] = None  #: Reward which is returned if the task completed with out an error. If this is set this reward will overwrite any other reward this function will generate. (Implementation specific might differ for child classes see documentation). If *None* no default reward is returned when task is finished. Defaults to *None*.
    run_on_reset: bool = True   #: This boolean can disable the running of the task when a reset is performed.
    additional_observations: Union[ObservationDefinition, ObservationDefinitionMulti, List[ObservationDefinition], None] = None  #: How many additional observations does this object yield. For np.ndarrays the number represents the size of the flattened array. Checking for correct size is the responsibility of the child classes.

    _first_time_setup_not_done: bool = PrivateAttr(True)    #: Whether or not the first time setup still needs to be done.

    @validator("additional_observations", pre=True)
    def validate_additional_observations(cls, v: str, values: Dict[str, Any]) -> Union[ObservationDefinition, ObservationDefinitionMulti, List[ObservationDefinition]]:
        if not isinstance(v, dict):
            if not int(v) == 0:
                v = {"name": f"unnamed_{str(uuid4())}", "value_min": -10, "value_max": 10, "observation_shape": [int(v)], "value_type": "float", "save_location": values["save_location"]}
            else:
                v = None
        return v


    def get_number_of_observations(self) -> Union[ObservationDefinition, ObservationDefinitionMulti, List[ObservationDefinition], None]:
        """Returns the number of observations which are generated by this SPORStep.

        Returns:
            Union[ObservationDefinition, ObservationDefinitionMulti, List[ObservationDefinition]]: Additional observation definitions
        """
        return self.additional_observations

    def get_default_observation(self, observations: ObservationType):
        if self.additional_observations is None:
            pass # observation do not need to be changed
        elif isinstance(self.additional_observations, list):
            for item in self.additional_observations:
                observations[item.name] = item.get_default_observation()
        else:
            observations[self.additional_observations.name] = self.additional_observations.get_default_observation()
        return observations

    def run(self, step_information: StepReturnType, environment_id: UUID4, validation_id: Optional[int] = None, core_count: int = 1, reset: bool = False) -> StepReturnType:
        """This function is called to complete the defined step of the current SPORObject. This functions needs to be overloaded by all child classes.

        Note:
            Please use these default values for the return values: observation = None, reward = 0, done = False, info = dict()

        Args:
            step_information (StepReturnType): Previously collected step values. Should be up-to-date.
            environment_id (UUID4): Environment ID which is used to distinguish different environments which run in parallel.
            validation_id (Optional[int], optional): During validation the validation id signifies the validation episode. If none the current episode is not a validation episode. Defaults to None.
            core_count (int, optional): Wanted core_count of the task to run. Defaults to 1.
            reset (bool, optional): Boolean on whether or not this object is called because of a reset. Defaults to False.


        Returns:
            StepReturnType: The full compliment of the step information are returned. The include observation, reward, done, info. 
        """
        return NotImplementedError

class SPORObjectCommandLine(SPORObject):
    """
        Base class for all possible SPOR object classes which use the command line. This class is meant to add some default functionality, so that users do not need to add them themselfs. These are:

        1. access to the (MPI) Multi-processors via :py:class:`SbSOvRL.spor.MultiProcessor`
        2. command line command
        3. additional flags
        4. working directory
        5. whether or not to use the :ref:`SPOR Communication Interface <sporcominterface>`

    """
    multi_processor: Optional[Union[MultiProcessor, MPIClusterMultiProcessor]] = None   #: Definition of the multi-processor. Defaults to *None*.
    use_communication_interface: bool = False   #: whether or not to use the SPOR communication interface. Defaults to *False*.
    add_step_information: bool = False #: whether or not to add the step information to the SPOR communications interface commandline options
    working_directory: str  #: Path to the directory in which the program should run in helpful for programs using relative paths. When {} appear in the string a UUID will be inserted for multiprocessing purposes.
    execution_command: str  #: Command which calls the program/script for this SPOR step.  
    command_options: List[str] = [] #: Command line options to add to the execution command. These do not include those from the communication interface these will be, if applicable, added separately.

    # communication_interface_variables
    _run_id: UUID4 = PrivateAttr(default=None)  #: If the communication interface is used this UUID will be used to identify the job/correct worker
    _run_func: Optional[Any] = PrivateAttr(default=None)  #: 
    _run_logger: Optional[Any] = PrivateAttr(default=None)  #: 

    @validator("working_directory")
    def validate_working_directory_path(cls, v: str):
        """Checks if the given path points to a correct directory. If a {} is present in the path only the path stub before this substring is validated. This functionality can be used for multiprocessing if each process needs to be in its own folder.

        Args:
            v (str): Object to validate

        Raises:
            SbSOvRLParserException: If path is not correct, this error is thrown.

        Returns:
            str: original path if no validation error occurs.
        """
        # check if placeholder for multiprocessing is available if not use v to validate directory path
        path = pathlib.Path(v.split("{}")[0]) if "{}" in v else pathlib.Path(v)
        if not (path.is_dir() and path.exists()):
            raise SbSOvRLParserException("SPORObjectCommandline", "unknown", f"The work_directory path {v} is a valid directory. When using multiprocessing placeholder please make sure that the directory before the placeholder is valid.")
        return v

    @validator("add_step_information")
    def validate_add_step_information_only_true_if_also_spor_com_is_true(cls, v: str, values: Dict[str, Any]):
        """Check that if the validated variable is true the value of the variable ``use_communication_interface`` is also true.

        Args:
            v (str): Object to validate
            v (str): Already validated values

        Raises:
            SbSOvRLParserException: If path is not correct, this error is thrown.

        Returns:
            str: value
        """
        if v:
            if "use_communication_interface" in values and values["use_communication_interface"]:
                pass
            else:
                raise SbSOvRLParserException("SPORObjectCommandline", "add_step_information", f"Please only set the add_step_information variable to True if the variable use_communication_interface is also True.")
        return v

    def get_multiprocessing_prefix(self, core_count: int) -> str:
        """Function which tries to get the correct multiprocessing command if available else an empty string is returned.

        Args:
            core_count (int): Maximal number of cores the multiprocessor can use.

        Returns:
            str: command string for the multi processor
        """
        ret_str = ""
        if self.multi_processor:
            ret_str = self.multi_processor.get_command(core_count)
        return ret_str


    def spor_com_interface(self, reset: bool, environment_id: UUID4, validation_id: Optional[int], step_information: StepReturnType) -> List[str]:
        """Generates the additional command line argument options used for the spor com interface.

        Args:
            reset (bool): Boolean on whether or not this object is called because of a reset.
            validation_id (Optional[int]): During validation the validation id signifies the validation episode. If none the current episode is not a validation episode.
            step_information (StepReturnType): Previously collected step values. Should be up-to-date.

        Returns:
            List[str]: List of additional command line options.
        """
        # return empty list if the com interface is not used
        if not self.use_communication_interface:
            return []
        # initialize run id if not already existing
        if not self._run_id:
            self._run_id = uuid4()

        # collect all parts of the communication interface
        interface_options: List[str] = ["--run_id", str(self._run_id)]
        interface_options.extend(["--base_save_location", str(self.save_location)])
        interface_options.extend(["--environment_id", str(environment_id)])
        if reset:
            interface_options.append("--reset")
        if validation_id is not None:
            interface_options.extend(["--validation_value", str(validation_id)])
        if self.add_step_information:
            json_step_information = {
                "observations": step_information[0],
                "reward": step_information[1],
                "done": step_information[2],
                "info": step_information[3]
            }
            # print(json_step_information)
            interface_options.extend(["--json_object", "'"+json.dumps(json_step_information, cls=SbSOvRL_JSONEncoder)+"'"])
        return interface_options
    
    def spor_com_interface_read(self, output: bytes, step_dict: Dict):
        """Interprets in the printed values from the spor com interface and will add them to the observation, reward, done and info return values.

        Args:
            output (bytes): Output of the command line call. To be interpreted.
            step_dict (Dict): Already known values of the step return this dict will be updated in place.
        """
        try: 
            returned_step_dict = literal_eval(output.decode(sys.getdefaultencoding()))
        except SyntaxError as err:
            self.get_logger().warning(f"An error was thrown while reading in the spor_com_interface return values of the step {self.name}. Please check that only the dictionary holding the correct information is printed during the execution of the external program.", exc_info=True)
            raise err
        self.spor_com_interface_add(returned_step_dict, step_dict)

    def spor_com_interface_add(self, returned_step_dict: Dict[str, Any], step_dict: Dict[str, Any]):
        """Add thenewly returned values to the observation, reward, done and info return values.

        Args:
            returned_step_dict (Dict): Newly received values.
            step_dict (Dict): Already known values of the step return this dict will be updated in place.
        """
        # add observations
        if self.additional_observations is None:
            pass # observation do not need to be changed
        elif isinstance(self.additional_observations, list):
            for item in self.additional_observations:
                step_dict["observation"][item.name] = returned_step_dict["observations"][item.name]
        else:
            step_dict["observation"][self.additional_observations.name] = np.array(returned_step_dict["observations"])
        # step_dict["observation"] = join_observations(step_dict["observation"], returned_step_dict["observations"], self.logger_name, self.additional_observations)
        
        # print(step_dict["observation"], step_dict["observation"].shape)
        join_infos(step_dict["info"][self.name], returned_step_dict["info"], self.logger_name)
        if step_dict["info"][self.name].get("reset_reason"):
            step_dict["info"]["reset_reason"] = step_dict["info"][self.name].get("reset_reason")
        step_dict["done"] = step_dict["done"] or returned_step_dict["done"]
        step_dict["reward"] = returned_step_dict["reward"]


    def run(self, step_information: StepReturnType, environment_id: UUID4, validation_id: Optional[int] = None, core_count: int = 1, reset: bool = False) -> StepReturnType:
        """This function runs the defined command line command with the defined arguments adding if necessary multi-processing flags and the spor communication interface.
        
        Args:
            step_information (StepReturnType): Previously collected step values. Should be up-to-date.
            environment_id (UUID4): Environment ID which is used to distinguish different environments which run in parallel.
            validation_id (Optional[int], optional): During validation the validation id signifies the validation episode. If none the current episode is not a validation episode. Defaults to None.
            core_count (int, optional): Wanted core_count of the task to run. Defaults to 1.
            reset (bool, optional): Boolean on whether or not this object is called because of a reset. Defaults to False.

        Returns:
            StepReturnType: The full compliment of the step information are returned. The include observation, reward, done, info. 
        """
        if self._first_time_setup_not_done:
            self.get_logger().info(f"Setup is performed for step with name {self.name}. For environment with id {environment_id}.")
            # add environemt_id to working path if necessary
            if "{}" in self.working_directory:
                self.working_directory = self.working_directory.format(environment_id)
                self.get_logger().info(f"Found placeholder for step with name {self.name}. New working directory is {self.working_directory}")

                # create run folder if necessary
                path = pathlib.Path(self.working_directory).expanduser().resolve()
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)

                if self.name == "main_solver":
                    shutil.copyfile(path.parent/"xns_multi.in", path/"xns.in")
                    self.get_logger().info(f"Copying file please.....")
            if len(self.command_options) == 1:
                possible_file = pathlib.Path(self.command_options[0]).resolve().absolute()
                if possible_file.exists() and possible_file.is_file() and possible_file.suffix == ".py":
                    ret_path = shutil.copy(possible_file, self.save_location/possible_file.name)
                    self.get_logger().debug(f"Successfully copied the python file {str(ret_path)}.")

                    try:
                        sys.path.insert(0, f'{possible_file.parent}{os.sep}')
                        func = importlib.import_module(possible_file.stem)
                    except ModuleNotFoundError as err:
                        self.get_logger().warning(f"Could not load the python file at {str(possible_file)}. This means the python file will be run externally. The runtime will be longer.")
                    else:
                        try:
                            self._run_func = func.main
                        except AttributeError as err:
                            self.get_logger().warning(f"Could not get main function from python file {str(possible_file)}. This means the python file will be run externally. The runtime will be longer.")
                        else:
                            self._run_logger = set_up_logger(f"spor_step_logger_{self.name.replace(' ','_')}", pathlib.Path(self.save_location/"logging"), VerbosityLevel.INFO, console_logging=False)
                            self.get_logger().info(f"Initialized internal python function calling for step {self.name}.")

            self._first_time_setup_not_done = False
        # first set up done
        
        env_logger = self.get_logger()
        step_return: Dict = {
            "observation": step_information[0], 
            "reward": step_information[1],
            "done": step_information[2],
            "info": step_information[3]
            }
        if reset and not self.run_on_reset: # skips the step without doing anything if reset and not run_on_reset add observations if the skiped step has some
            if self.additional_observations is not None:
                step_return["observation"] = self.get_default_observation(step_return["observation"])
        else:   # executes the step
            multi_proc_prefix = self.get_multiprocessing_prefix(core_count=core_count)
            command_list = [multi_proc_prefix, self.execution_command, *self.command_options, *self.spor_com_interface(reset, environment_id, validation_id, step_information)]
            if self._run_func is not None:
                args = spor_com_parse_arguments(command_list[3:])
                exit_code = 0
                output = None
                try:
                    current_dir = os.getcwd()
                    os.chdir(self.working_directory)
                    output = self._run_func(args, self._run_logger)
                    os.chdir(current_dir)
                except Exception as err:
                    self.get_logger().warning(f"Could not run the internalized python spor function without error. The follwing error was thrown {err}. Please check if this is a user error.")
                    exit_code = 404
            else:
                command = " ".join(command_list)
                exit_code, output = call_commandline(command, self.working_directory, env_logger)
            step_return["info"][self.name] = {
               "output": output,
               "exit_code": int(exit_code)
            }
            if exit_code != 0:
                env_logger.info("SPOR Step thrown error.")
                if self.stop_after_error:
                    env_logger.info("Will stop episode now, due to thrown error.")
                    step_return["done"] = True
                    if self.name == "main_solver" and int(exit_code) == 137:    # exit code for xns if mesh is tangled hopefully
                        step_return["info"]["reset_reason"] = f"meshTangled-{self.name}"
                    elif self.name == "main_solver" and not output:   # the main_solver should always have an output only time no output is generated srun aborted early (hopefully)
                        step_return["info"]["reset_reason"] = f"srunError-{self.name}"
                        self.get_logger().warning("Could not find any output of failed command assuming srun/mpi error. Trying to exit training now.")
                    else:
                        step_return["info"]["reset_reason"] = f"ExecutionFailed-{self.name}"
                step_return["reward"] = self.reward_on_error
                if self.additional_observations is not None:
                    step_return["observation"] = self.get_default_observation(step_return["observation"])
            else:
                if self.use_communication_interface:
                    if isinstance(output, dict):
                        self.spor_com_interface_add(output, step_return)
                    else:
                        self.spor_com_interface_read(output, step_return)
                if self.reward_on_completion:
                    if step_return["reward"]:
                        env_logger.warning(f"A reward {step_return['reward']} is already set but a default reward {self.reward_on_completion} on completion is also set. The default reward will overwrite the already set reward. Please check if this is the wanted behavior, if not set the reward on completion to None.")
                    step_return["reward"] = self.reward_on_completion

        return step_return["observation"], step_return["reward"], step_return["done"], step_return["info"]

SPORObjectTypes = SPORObjectCommandLine
           
class SPORList(SbSOvRL_BaseModel):
    """
    The SPORList defines the steps which need to be taken for after the FFD is done. These can include the fluid solver, post-processing steps, reward generation, additional logging, etc. 

    If a step sends a stop signal all consecutive steps will be ignored and the 
    Each step can generate observations, rewards and info for the current episode step.  These are aggregated as follows.
    """
    steps: List[SPORObjectTypes]  #: List of SPOR objects. Each SPOR object is a step in the environment.
    reward_aggregation: Literal["sum", "min", "max", "mean", "minmax"]   #: Definition on how the reward should be aggregated. Definition for each option is given :doc:`here <spor>`.

    _rewards: List[RewardType] = PrivateAttr(default_factory=list)  #: internal data holder for the rewards collected during a single episode. Could be local variable but is object variable to have to option to read it out later on, if need be.
    # _observations: Any = PrivateAttr(default=None)
    # _info: InfoType = PrivateAttr(default_factory=dict)

    def get_number_of_observations(self) -> Optional[List[Union[ObservationDefinition, ObservationDefinitionMulti]]]:
        """Aggregates the observations of all steps and returns the flattened list of observations spaces comming from the SPORSteps.

        Returns:
            int: Number of observations.
        """
        observations = []
        for step in self.steps:
            fi = step.get_number_of_observations()
            if fi is None:
                continue
            if isinstance(fi, list):
                observations.extend(fi)
            else:
                observations.append(fi)
        if len(observations) == 0:
            return None
        return observations

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
                self.get_logger().error(err_mesg)
                raise RuntimeError(err_mesg)
        else:
            self.get_logger().warning("No rewards given. Setting reward to zero (0). Please check if reward should be given but did not register.")
            return 0

    def run(self, step_information: StepReturnType, environment_id: UUID4, validation_id: Optional[int] = None, core_count: int = 1, reset: bool = False) -> StepReturnType:
        """Runs through all steps of the list and handles each output. When done all received observations and infos are jointly returned with the reward and the indicator whether or not the episode should be terminated.

        The values are added in-place to the step_information variable but is also returned.

        Args:
            step_information (StepReturnType): Previously collected step values. Should be up-to-date.
            environment_id (UUID4): Environment ID which is used to distinguish different environments which run in parallel.
            validation_id (Optional[int], optional): If currently in validation the validation needs to be supplied for steps which need this. If none given no validation is currently performed. Defaults to None.
            core_count (int, optional): Wanted core_count of the task to run. Defaults to 1.
            reset (bool, optional): Boolean on whether or not this object is called because of a reset. Defaults to False.

        Returns:
            StepReturnType: Results of all steps.
        """
        self._rewards = []
        observations, reward, done, info = step_information
        for step in self.steps:
            start = timer()
            if done:
                if step.additional_observations:
                    # print("adding zeros")
                    observations = step.get_default_observation(observations)
                    # print(observations, observations.shape)
            else:
                if reset and not (step.run_on_reset):   # ignore if step should be skiped during reset procedure
                    continue
                try:
                    observations, reward, done, info = step.run(step_information, environment_id, validation_id, core_count, reset)
                    if reward is not None:
                        self._rewards.append(reward)
                    # if info:
                    #     join_infos(info, info, self.logger_name)
                except Exception as exp:
                    self.get_logger().warning(f"The current step with name {step.name} has thrown an error: {exp}.", exc_info=1)
                    if step.stop_after_error:
                        done = True
                        self.get_logger().warning(f"Due to the error in the step {step.name} this episode will now be terminated.")
            if done:
                if not info.get("reset_reason"):
                    info["reset_reason"] = "unspecified"
                    self.get_logger().warning(f"Step {step.name} resulted in episode end but did not specify reason for termination.")
            reward = self._compute_reward()
            step_information = (observations, reward, done, info)
            end = timer()
            self.get_logger().debug(f"SPOR Step {step.name} took {end-start} seconds.")
        return step_information
