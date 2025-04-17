"""SPOR definition Read Carefully.

Solver Postprocessing Observation Reward (SPOR) represents all steps which

follow after completing Geometry adaption, where the geometry is
deformed via the actions.

This part of the ReLeSO environment definition is a list of steps the
environment will run through in series. Each step can be configured to use the
ReLeSO communication protocol.

A more in depth documentation of the SPOR concept is given here
:ref:`SPOR Communication Interface <sporcominterface>`
"""

import importlib
import json
import os
import pathlib
import shutil
import sys
import traceback
from ast import literal_eval
from timeit import default_timer as timer
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

import numpy as np
from pydantic import UUID4, Field, conint, validator
from pydantic.fields import PrivateAttr

from releso.base_model import BaseModel
from releso.exceptions import ParserException
from releso.observation import (
    ObservationDefinition,
    ObservationDefinitionMulti,
)
from releso.util.logger import VerbosityLevel, set_up_logger
from releso.util.reward_helpers import spor_com_parse_arguments
from releso.util.types import ObservationType, RewardType, StepReturnType
from releso.util.util_funcs import JSONEncoder, call_commandline, join_infos


class MultiProcessor(BaseModel):
    """Definition of the utilized multiprocessor. Default MPIEXEC.

    The MultiProcessor gives access to command prefixes which can enable
    parallelization. Currently only tested for MPI.

    Note:
        The cluster uses PATH variables to correctly call mpi. IF you want to
        use MPI multiprocessing on the cluster please use
        :py:class:`ReLeSO.spor.MPIClusterMultiProcessor`.
    """

    #: command prefix which is needed to parallelize the give task
    command: str = "mpiexec -n"
    #: Max core count to use for this parallelization option. This variable
    #: is not shared between tasks or environments so the user has to split
    #: these up before hand.
    max_core_count: conint(ge=1)
    #: Whether or not to always use the multiprocessing option. This is
    #: helpful if multiprocessing command prefix is needed even if core count
    #: is 1. Defaults to *False*.
    always_use: bool = False

    def get_command(self, core_count: int) -> str:
        """Return the string that represents the call for multiprocessing.

        Args:
            core_count (int): Number of wanted cores.

        Returns:
            str: string representing the command line call prefix
        """
        resulting_core_count = max(1, min(core_count, self.max_core_count))
        return (
            f"{self.command} {resulting_core_count}"
            if resulting_core_count > 1 or self.always_use
            else ""
        )


class MPIClusterMultiProcessor(MultiProcessor):
    """Multi-processor extension for the cluster MPI call."""

    #: The location title needs to be set to ``cluster`` so that
    location: Literal["cluster"]
    # : command prefix which is needed to parallelize the give task

    command: str = "$MPIEXEC"
    #: additional flags for the mpi call.
    mpi_flags_variable: str = "$FLAGS_MPI_BATCH"

    def get_command(self, core_count: int) -> str:
        """Return the string that represents the call for multiprocessing.

        Note:
            Since the cluster uses PATH variables for its computation. This
            function will read these in and use them correspondingly. But will
            not change them. This limitation is done so that there is less
            interference during Multi-Environment training.

        Args:
            core_count (int): Number of wanted cores.

        Raises:
            RuntimeError: This error is thrown if the mpi_flags PATH variable
            is not of the shape expected by the function. Expected are 2 parts
            -np [core_count] where -np can be anything and the core_count
            should be a number.

        Returns:
            str: string representing the command line call prefix
        """
        # self.get_logger().warning("Using Cluster mpi")
        # check if environment variable for mpi has the correct core number.
        resulting_core_count = max(1, min(core_count, self.max_core_count))
        if resulting_core_count == 1 and not self.always_use:
            return ""
        environ_mpi_flags: List = str(
            os.environ[self.mpi_flags_variable.replace("$", "")]
        ).split()
        local_mpi_flags = self.mpi_flags_variable
        if len(environ_mpi_flags) == 0:
            core_qualifier = "-n"
            set_core_count = -1
        elif len(environ_mpi_flags) == 2:
            core_qualifier, set_core_count = environ_mpi_flags
        else:
            self.get_logger().error(
                "The environment variable for mpi cluster did not look as "
                "expected. Please use standard MultiProcessor or revise the "
                "code. The variable is as follow "
                f"{environ_mpi_flags, len(environ_mpi_flags)}"
            )
            raise RuntimeError(
                "MPI Cluster environment variable not as expected. "
                "More information in the log."
            )
        if set_core_count != resulting_core_count:
            local_mpi_flags = f"{core_qualifier} {resulting_core_count}"
        self.get_logger().debug(
            f"Using Cluster mpi with command "
            f"{os.environ[self.command.replace('$', '')]} and additional "
            f"flags {local_mpi_flags}"
        )
        return f"{self.command} {local_mpi_flags}"


class SPORObject(BaseModel):
    """Base class SPORObject can not be instantiated.

    Base class for all possible SPOR object classes. Theses objects
    represent the possible steps that can be/are executed after the FFD
    finished.

    The following arguments need to be set for SPOR objects, further
    parameters are use-case specific.
    """

    #: name of the SPOR step. Makes so that it is possible to distinguish
    #: different steps more easily.
    name: str
    #: Whether to stop after this step if this step has thrown an error.
    #: This is important if
    stop_after_error: bool = True
    #: Reward which is returned if any kind of error is thrown during
    #: completing the defined task.
    reward_on_error: float
    #: Reward which is returned if the task completed with out an error. If
    #: this is set this reward will overwrite any other reward this function
    #: will generate. (Implementation specific might differ for child classes
    #: see documentation). If *None* no default reward is returned when task is
    #: finished. Defaults to *None*.
    reward_on_completion: Optional[float] = None
    #: This boolean can disable the running of the task when a reset is
    #: performed.
    run_on_reset: bool = True
    #: How many additional observations does this object yield. Please define
    #: this correctly and not just write in a single number. Be responsible,
    #: the single number functionality is only in here due to backwards
    #: compatibility reasons.
    additional_observations: Union[
        ObservationDefinition,
        ObservationDefinitionMulti,
        List[Union[ObservationDefinition, ObservationDefinitionMulti]],
        None,
    ] = None

    #: Whether or not the first time setup still needs to be done.
    _first_time_setup_not_done: bool = PrivateAttr(True)

    @validator("additional_observations", pre=True)
    @classmethod
    def validate_additional_observations(
        cls, v: str, values: Dict[str, Any]
    ) -> Union[
        ObservationDefinition,
        ObservationDefinitionMulti,
        List[Union[ObservationDefinition, ObservationDefinitionMulti]],
    ]:
        """Validator additional_observations.

        Validates the the additional observations variable. This is a pre
        validation function only used for back compatibility reasons.

        Args:
            v (str): See pydantic
            values (Dict[str, Any]):  See pydantic

        Returns:
            Union[
                ObservationDefinition, ObservationDefinitionMulti,
                List[ObservationDefinition]]: Pre validated values.
        """
        if not isinstance(v, (dict, list)):
            if not int(v) == 0:
                # We might want to deprecate the hidden option to instantiate
                #  additional observations via an int. Building up a default
                #  observation definition.
                # TODO: change name to something more meaningful for example
                # using the function name values["name"]
                v = {
                    "name": f"unnamed_{str(uuid4())}",
                    "value_min": -1,
                    "value_max": 1,
                    "observation_shape": [int(v)],
                    "value_type": "float",
                    "save_location": values["save_location"],
                }

            else:
                v = None

        return v

    def get_observations(
        self,
    ) -> Union[
        ObservationDefinition,
        ObservationDefinitionMulti,
        List[Union[ObservationDefinition, ObservationDefinitionMulti]],
        None,
    ]:
        """Return number of observation the step generates.

        Returns the number of observations which are generated by this
        SPORStep.

        Returns:
            Union[
                ObservationDefinition, ObservationDefinitionMulti,
                List[ObservationDefinition]]: Additional observation
                definitions
        """
        return self.additional_observations

    def get_default_observation(
        self, observations: ObservationType
    ) -> ObservationType:
        """Generate default observations if no were returned or step failed.

        Generates default observations for all additional observations defined
        in this object.

        Args:
            observations (ObservationType): Already defined observations to
            which the additional observations are added to.

        Returns:
            ObservationType: Observations with now the additional observations
            of this object added (values used are the default values)
        """
        if self.additional_observations is None:
            pass  # observation do not need to be changed
        elif isinstance(self.additional_observations, list):
            for item in self.additional_observations:
                observations[item.name] = item.get_default_observation()
        else:
            observations[self.additional_observations.name] = (
                self.additional_observations.get_default_observation()
            )
        return observations

    def run(
        self,
        step_information: StepReturnType,
        environment_id: UUID4,
        validation_id: Optional[int] = None,
        core_count: int = 1,
        reset: bool = False,
    ) -> StepReturnType:
        """Performs the defined step.

        This function is called to complete the defined step of the current
        SPORObject. This functions needs to be overloaded by all child classes.

        Note:
            Please use these default values for the return values: observation
            = None, reward = 0, done = False, info = dict()

        Args:
            step_information (StepReturnType): Previously collected step
            values. Should be up-to-date.
            environment_id (UUID4): Environment ID which is used to distinguish
            different environments which run in parallel.
            validation_id (Optional[int], optional): During validation the
            validation id signifies the validation episode. If none the current
            episode is not a validation episode. Defaults to None.
            core_count (int, optional): Wanted core_count of the task to run.
            Defaults to 1.
            reset (bool, optional): Boolean on whether or not this object is
            called because of a reset. Defaults to False.


        Returns:
            StepReturnType:
                The full compliment of the step information are returned.
                The include observation, reward, done, info.
        """
        raise NotImplementedError


class SPORObjectExecutor(SPORObject):
    """Base definition of a SPORCommandLineObject. Can be instantiated.

    Base class for all possible SPOR object classes which use the command
    line. This class is meant to add some default functionality, so that
    users do not need to add them themselves. These are:

    1. access to the (MPI) Multi-processors via
        :py:class:`ReLeSO.spor.MultiProcessor`
    2. command line command
    3. additional flags
    4. working directory
    5. whether or not to use the
        :ref:`SPOR Communication Interface <sporcominterface>`

    """

    #: Definition of the multi-processor. Defaults to *None*.
    multi_processor: Optional[
        Union[MultiProcessor, MPIClusterMultiProcessor]
    ] = None
    #: whether or not to use the SPOR communication interface.
    #: Defaults to *False*.
    use_communication_interface: bool = False
    # : whether or not to add the step information to the SPOR communications
    #: interface commandline options including the observations, info, done,
    #: reward
    add_step_information: bool = False
    #: Path to the directory in which the program should run, helpful for
    #: programs using relative paths. If `{}` are the first characters of the
    #: path, they will be replaced by the save_location of the current run. If
    #: `{}` are present in the path, the current environment id will be used to
    #: replace the placeholder. This is necessary for multiprocessing.
    working_directory: str

    # communication_interface_variables
    #: If the communication interface is used this UUID will be used to
    #: identify the job/correct worker
    _run_id: UUID4 = PrivateAttr(default=None)

    @validator("working_directory")
    @classmethod
    def validate_working_directory_path(cls, v: str):
        """Validator working_directory.

        Checks if the given path points to a correct directory. If a {} is
        present in the path only the path stub before this substring is
        validated. This functionality can be used for multiprocessing if each
        process needs to be in its own folder.

        Args:
            v (str): Object to validate

        Raises:
            ParserException: If path is not correct, this error is
            thrown.

        Returns:
            str: original path if no validation error occurs.
        """
        # check if placeholder for multiprocessing is available if not use
        # v to validate directory path
        if not v.startswith("{}"):
            path = (
                pathlib.Path(v.split("{}")[0])
                if "{}" in v
                else pathlib.Path(v)
            )
            if not (path.is_dir() and path.exists()):
                raise ParserException(
                    "SPORObjectCommandline",
                    "unknown",
                    f"The work_directory path {v} does not exist or is not a"
                    " valid directory."
                    " If you are using the multiprocessing placeholder {}, make "
                    "sure, that the path before the placeholder is valid and "
                    "exists.",
                )
        return v

    @validator("add_step_information")
    @classmethod
    def validate_add_step_information_only_true_if_also_spor_com_is_true(
        cls, v: str, values: Dict[str, Any]
    ):
        """Validator add_step_information.

        Check that if the validated variable is true the value of the variable
        ``use_communication_interface`` is also true.

        Args:
            v (str): Object to validate
            values (str): Already validated values

        Raises:
            ParserException: If path is not correct, this error is thrown.

        Returns:
            str: value
        """
        if v:
            if (
                "use_communication_interface" in values
                and values["use_communication_interface"]
            ):
                pass
            else:
                raise ParserException(
                    "SPORObjectCommandline",
                    "add_step_information",
                    "Please only set the add_step_information variable to "
                    "True if the variable use_communication_interface is also "
                    "True.",
                )
        return v

    def setup_working_directory(self, environment_id: UUID4):
        """Set up the working directory for the SPORStep.

        Args:
            environment_id (UUID4): Id of the environment. This is the working
                directory.
        """
        if self.working_directory.startswith("{}"):
            self.working_directory = self.working_directory.format(
                self.save_location, environment_id
            )
            pathlib.Path(self.working_directory).expanduser().resolve().mkdir(
                parents=True, exist_ok=True
            )
            self.get_logger().info(
                f"Found placeholder for step with name {self.name}. New "
                f"working directory is {self.working_directory}. Directory was created."
            )
        elif "{}" in self.working_directory:
            self.working_directory = self.working_directory.format(
                environment_id
            )

            # create run folder if necessary
            path = pathlib.Path(self.working_directory).expanduser().resolve()
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.get_logger().info(
                    f"Found placeholder for step with name {self.name}. New "
                    f"working directory is {self.working_directory}. "
                    "Directory was created."
                )
            else:
                self.get_logger().info(
                    f"Found placeholder for step with name {self.name}. New "
                    f"working directory is {self.working_directory}. "
                    "Directory already exists, step will use it."
                )
            # TODO this is a XNS specific case
            if self.name == "main_solver":  # pragma: no cover
                shutil.copyfile(path.parent / "xns_multi.in", path / "xns.in")
                self.get_logger().info("Copying file.....")

    def get_multiprocessing_prefix(self, core_count: int) -> str:
        """Add commandline prefix for mpi multiprocessor.

        Function which tries to get the correct multiprocessing command if
        available else an empty string is returned.

        Args:
            core_count (int): Maximal number of cores the multiprocessor can
            use.

        Returns:
            str: command string for the multi processor
        """
        ret_str = ""
        if self.multi_processor:
            ret_str = self.multi_processor.get_command(core_count)
        return ret_str

    def spor_com_interface(
        self,
        reset: bool,
        environment_id: UUID4,
        validation_id: Optional[int],
        step_information: StepReturnType,
    ) -> List[str]:
        """Add spor com interface command line options.

        Generates the additional command line argument options used for the
        spor com interface.

        Args:
            reset (bool): Boolean on whether or not this object is called
                because of a reset.
            validation_id (Optional[int]): During validation the validation id
                signifies the validation episode. If none the current episode
                is not a validation episode.
            step_information (StepReturnType): Previously collected step
                values. Should be up-to-date.
            environment_id (UUID4): Environment ID which is used to distinguish
                different environments which run in parallel.

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
        interface_options.extend([
            "--base_save_location",
            str(self.save_location),
        ])
        interface_options.extend(["--environment_id", str(environment_id)])
        if reset:
            interface_options.append("--reset")
        if validation_id is not None:
            interface_options.extend([
                "--validation_value",
                str(validation_id),
            ])
        if self.add_step_information:
            json_step_information = {
                "observations": step_information[0],
                "reward": step_information[1],
                "done": step_information[2],
                "info": step_information[3],
            }
            # print(json_step_information)
            interface_options.extend([
                "--json_object",
                "'" + json.dumps(json_step_information, cls=JSONEncoder) + "'",
            ])
        return interface_options

    def spor_com_interface_read(self, output: bytes, step_dict: Dict):
        """Read in return values of the spor com interface.

        Interprets in the printed values from the spor com interface and will
        add them to the observation, reward, done and info return values.

        Args:
            output (bytes): Output of the command line call. To be interpreted.
            step_dict (Dict): Already known values of the step return this dict
            will be updated in place.
        """
        try:
            returned_step_dict = literal_eval(
                output.decode(sys.getdefaultencoding())
            )
        except SyntaxError as err:
            self.get_logger().warning(
                f"An error was thrown while reading in the spor_com_interface "
                f"return values of the step {self.name}. Please check that "
                "only the dictionary holding the correct information is "
                "printed during the execution of the external program.",
                exc_info=True,
            )
            raise SyntaxError from err
        self.spor_com_interface_add(returned_step_dict, step_dict)

    def spor_com_interface_add(
        self, returned_step_dict: Dict[str, Any], step_dict: Dict[str, Any]
    ):
        """Add returned step information of the spor com interface.

        Add the newly returned values to the observation, reward, done and
        info return values.

        Args:
            returned_step_dict (Dict): Newly received values.
            step_dict (Dict): Already known values of the step return this dict
            will be updated in place.
        """
        # add observations
        if self.additional_observations is None:
            pass  # observation do not need to be changed
        elif isinstance(self.additional_observations, list):
            for item in self.additional_observations:
                step_dict["observation"][item.name] = returned_step_dict[
                    "observations"
                ][item.name]
        else:
            step_dict["observation"][self.additional_observations.name] = (
                np.array(returned_step_dict["observations"])
            )

        join_infos(
            step_dict["info"][self.name],
            returned_step_dict["info"],
            self.logger_name,
        )
        if value := step_dict["info"].get(self.name):
            if value := value.get("reset_reason"):
                step_dict["info"]["reset_reason"] = value
        step_dict["done"] = step_dict["done"] or returned_step_dict["done"]
        step_dict["reward"] = returned_step_dict["reward"]

    def run(
        self,
        step_information: StepReturnType,
        environment_id: UUID4,
        validation_id: Optional[int] = None,
        core_count: int = 1,
        reset: bool = False,
    ) -> StepReturnType:
        """Run method for the Execution types of SPORSteps.

        Abstract method.

        Args:
            step_information (StepReturnType): _description_
            environment_id (UUID4): _description_
            validation_id (Optional[int], optional): _description_.
                Defaults to None.
            core_count (int, optional): _description_. Defaults to 1.
            reset (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            StepReturnType: _description_
        """
        raise NotImplementedError


class SPORObjectPythonFunction(SPORObjectExecutor):
    """Base definition of a SPORObjectPythonFunction. Parent class.

    Base class for all possible SPOR python object classes which use
    internalized python function mechanism.
    This class is meant to add some default functionality, so that
    users do not need to add them themselves. These are:

    1. Let an internalized python function be run and save the function data
    2. working directory
    3. whether or not to use the
        :ref:`SPOR Communication Interface <sporcominterface>`

    """

    #: function handler used for python functions which are loaded inside the
    #: program envelop from the outside
    _run_func: Optional[Any] = PrivateAttr(default=None)
    #: logger which is used for the internalized python functions
    _run_logger: Optional[Any] = PrivateAttr(default=None)
    #: persistent function data is not touched by releso
    _func_data: Optional[Any] = PrivateAttr(default=None)

    def run(
        self,
        step_information: StepReturnType,
        environment_id: UUID4,
        validation_id: Optional[int] = None,
        core_count: int = 1,
        reset: bool = False,
    ) -> StepReturnType:
        """This function loads and executes the defined python file.

        The herein defined python file needs to have a function called `main`
        with three parameters.

        1. args: Namespace see SPORCommInterface
        2. logger: logger to be used in the function
        3. func_data: data variable can be used to store persistent data


        Args:
            step_information (StepReturnType):
                Previously collected step values. Should be up-to-date.
            environment_id (UUID4):
                Environment ID which is used to distinguish different
                environments which run in parallel.
            validation_id (Optional[int], optional):
                During validation the validation id signifies the validation
                episode. If none the current episode is not a validation
                episode. Defaults to None.
            core_count (int, optional):
                Can only be one since no multi processing is possible in this
                case.
            reset (bool, optional):
                Boolean on whether or not this object is called because of a
                reset. Defaults to False.

        Returns:
            StepReturnType:
                The full compliment of the step information are returned.
                The include observation, reward, done, info.
        """
        env_logger = self.get_logger()
        step_return: Dict = {
            "observation": step_information[0],
            "reward": step_information[1],
            "done": step_information[2],
            "info": step_information[3],
        }
        # skips the step without doing anything if reset and not run_on_reset
        # add observations if the skipped step has some
        if reset and not self.run_on_reset:
            if self.additional_observations is not None:
                step_return["observation"] = self.get_default_observation(
                    step_return["observation"]
                )
        else:  # executes the step
            args = spor_com_parse_arguments(
                self.spor_com_interface(
                    reset=reset,
                    environment_id=environment_id,
                    validation_id=validation_id,
                    step_information=step_information,
                )
            )
            exit_code = 0
            output = None
            current_dir = os.getcwd()
            try:
                os.chdir(self.working_directory)
                output, self._func_data = self._run_func(
                    args, self._run_logger, self._func_data
                )
            except Exception as err:  # noqa: BLE001
                self.get_logger().warning(
                    f"Could not run the internalized python spor function "
                    f"without error. The following error was thrown {err}."
                    " Please check if this is a user error."
                    f"Traceback is {traceback.format_exc()}."
                )
                exit_code = 404
            os.chdir(current_dir)

            step_return["info"][self.name] = {
                "output": output,
                "exit_code": int(exit_code),
            }
            self.get_logger().debug(
                f"The return step return of the current "
                f"command is: {step_return}"
            )
            if exit_code != 0:  # error thrown during command execution
                env_logger.info(f"SPOR Step {self.name} thrown error.")
                if self.stop_after_error:
                    env_logger.info(
                        "Will stop episode now, due to thrown error."
                    )
                    step_return["done"] = True
                    step_return["info"]["reset_reason"] = (
                        f"ExecutionFailed-{self.name}"
                    )
                step_return["reward"] = self.reward_on_error
                if self.additional_observations is not None:
                    step_return["observation"] = self.get_default_observation(
                        step_return["observation"]
                    )
            else:
                if self.use_communication_interface:
                    if isinstance(output, dict):
                        self.spor_com_interface_add(output, step_return)
                    else:  # this should never be accessed with python funcs
                        raise RuntimeError(
                            "The output of the internalized python function "
                            "is not a dictionary. Please check the function "
                            "and the documentation."
                        )  # pragma: no cover
                        # self.spor_com_interface_read(output, step_return)
                if self.reward_on_completion:
                    if step_return["reward"]:
                        env_logger.warning(
                            f"A reward {step_return['reward']} is already set "
                            f"but a default reward {self.reward_on_completion}"
                            " on completion is also set. The default reward "
                            "will overwrite the already set reward. Please "
                            "check if this is the wanted behavior, if not set "
                            "the reward on completion to None."
                        )
                    step_return["reward"] = self.reward_on_completion

        return (
            step_return["observation"],
            step_return["reward"],
            step_return["done"],
            step_return["info"],
        )


class SPORObjectInternalPythonFunction(SPORObjectPythonFunction):
    """This class will load SPOR functions provided by this package.

    Currently available functions are the following:

        - xns_cnn: This function will use the result of the xns solver and
            create an image of the solution. The image is created via contourf.

    """

    function_name: Literal["xns_cnn"]

    #: function handler used for python functions which are loaded inside the
    #: program envelop from the outside
    _run_func: Optional[Any] = PrivateAttr(default=None)
    #: logger which is used for the internalized python functions
    _run_logger: Optional[Any] = PrivateAttr(default=None)
    #: persistent function data is not touched by releso
    _func_data: Optional[Any] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        """Internalized python function init function.

        Initializes the function and adds the observation data.

        Raises:
            ValueError: Requested function not found.
        """
        super().__init__(**kwargs)
        if self.function_name == "xns_cnn":
            from releso.util import cnn_xns_observations

            self._run_func = cnn_xns_observations.main
            self.additional_observations = (
                cnn_xns_observations.define_observation_definition()
            )
        else:  # pragma: no cover # Literal validation should catch this
            raise ValueError(
                f"The function {self.function_name} is unknown. Please check"
                " the spelling and version of the package."
            )

    def run(
        self,
        step_information: StepReturnType,
        environment_id: UUID4,
        validation_id: Optional[int] = None,
        core_count: int = 1,
        reset: bool = False,
    ) -> StepReturnType:
        """This function loads and executes the defined python file.

        The herein defined python file needs to have a function called `main`
        with three parameters.

        1. args: Namespace see SPORCommInterface
        2. logger: logger to be used in the function
        3. func_data: data variable can be used to store persistent data


        Args:
            step_information (StepReturnType):
                Previously collected step values. Should be up-to-date.
            environment_id (UUID4):
                Environment ID which is used to distinguish different
                environments which run in parallel.
            validation_id (Optional[int], optional):
                During validation the validation id signifies the validation
                episode. If none the current episode is not a validation
                episode. Defaults to None.
            core_count (int, optional):
                Can only be one since no multi processing is possible in this
                case.
            reset (bool, optional):
                Boolean on whether or not this object is called because of a
                reset. Defaults to False.

        Returns:
            StepReturnType:
                The full compliment of the step information are returned.
                The include observation, reward, done, info.
        """
        if self._first_time_setup_not_done:
            self.setup_working_directory(environment_id)
            self._first_time_setup_not_done = False

        return super().run(
            step_information, environment_id, validation_id, core_count, reset
        )


class SPORObjectExternalPythonFunction(SPORObjectPythonFunction):
    """Load a python function from an external file.

    This class is meant to load a python function from an external file and
    execute it. The function will be imported in to the package and run in the
    same python environment as the rest of the package is. If you want to call
    the function you can use the SPORCommandLineObject.

    The function needs to be called main and needs to have the
    following signature:

        args, logger, func_data

    Where args are the SPOR COMM arguments, logger is a logger provided by
    the SPORObject and, func_data is a persistent data variable which is not
    touched by the SPORObject.
    """

    python_file_path: Union[str, pathlib.Path]

    #: function handler used for python functions which are loaded inside the
    #: program envelop from the outside
    _run_func: Optional[Any] = PrivateAttr(default=None)
    #: logger which is used for the internalized python functions
    _run_logger: Optional[Any] = PrivateAttr(default=None)
    #: persistent function data is not touched by releso
    _func_data: Optional[Any] = PrivateAttr(default=None)

    @validator("python_file_path")
    @classmethod
    def validate_execution_command_path(cls, v: str):
        """Validator of the path to the python file.

        Check if path to execution command exist.

        Does not check if the file is correct or the function inside this file
        exists.

        TODO check if the correct function signature exists inside the file

        Args:
            v (str): Object to validate

        Raises:
            ParserException: If path is not correct, this error is thrown.

        Returns:
            pathlib.Path: original path if no validation error occurs.
        """
        # check if placeholder for multiprocessing is available if not use v
        # to validate directory path
        path = pathlib.Path(v)
        if path.exists() and path.is_file() and path.suffix == ".py":
            path = path.expanduser().resolve().as_posix()
        else:
            raise ParserException(
                "SPORObjectCommandline",
                "python_file_path",
                f"The path {v} does not exist.",
            )
        return path

    def run(
        self,
        step_information: StepReturnType,
        environment_id: UUID4,
        validation_id: Optional[int] = None,
        core_count: int = 1,
        reset: bool = False,
    ) -> StepReturnType:
        """This function loads and executes the defined python file.

        The herein defined python file needs to have a function called `main`
        with three parameters.

        1. args: Namespace see SPORCommInterface
        2. logger: logger to be used in the function
        3. func_data: data variable can be used to store persistent data


        Args:
            step_information (StepReturnType):
                Previously collected step values. Should be up-to-date.
            environment_id (UUID4):
                Environment ID which is used to distinguish different
                environments which run in parallel.
            validation_id (Optional[int], optional):
                During validation the validation id signifies the validation
                episode. If none the current episode is not a validation
                episode. Defaults to None.
            core_count (int, optional):
                Can only be one since no multi processing is possible in this
                case.
            reset (bool, optional):
                Boolean on whether or not this object is called because of a
                reset. Defaults to False.

        Returns:
            StepReturnType:
                The full compliment of the step information are returned.
                The include observation, reward, done, info.
        """
        if self._first_time_setup_not_done:
            self.python_file_path = pathlib.Path(self.python_file_path)
            self.setup_working_directory(environment_id)
            self.get_logger().info(
                f"Setup is performed for step with name {self.name}. For "
                f"environment with id {environment_id}."
            )

            ret_path = shutil.copy(
                self.python_file_path,
                self.save_location / self.python_file_path.name,
            )
            self.get_logger().debug(
                f"Successfully copied the python file {str(ret_path)}"
            )

            # trying to internalize the function
            try:
                sys.path.insert(0, f"{self.python_file_path.parent}{os.sep}")
                func = importlib.import_module(self.python_file_path.stem)
            except ModuleNotFoundError as err:
                raise RuntimeError(
                    f"Could not load the python file at "
                    f"{str(self.python_file_path)}"
                ) from err
            else:
                try:
                    self._run_func = func.main
                except AttributeError as err:
                    raise RuntimeError(
                        f"Could not get main function from python file"
                        f" {str(self.python_file_path)}."
                    ) from err
                else:
                    self._run_logger = set_up_logger(
                        f"spor_step_logger_{self.name.replace(' ', '_')}",
                        pathlib.Path(self.save_location / "logging"),
                        VerbosityLevel.INFO,
                        console_logging=False,
                    )
                    self.get_logger().info(
                        f"Initialized internal python function calling"
                        f" for step {self.name}."
                    )

            self._first_time_setup_not_done = False

        return super().run(
            step_information, environment_id, validation_id, core_count, reset
        )


class SPORObjectCommandLine(SPORObjectExecutor):
    """Base definition of a SPORCommandLineObject. Can be instantiated.

    Base class for all possible SPOR object classes which use the command
    line. This class is meant to add some default functionality, so that
    users do not need to add them themselves. These are:

    1. access to the (MPI) Multi-processors via
        :py:class:`ReLeSO.spor.MultiProcessor`
    2. command line command
    3. additional flags
    4. working directory
    5. whether or not to use the
        :ref:`SPOR Communication Interface <sporcominterface>`

    """

    #: Command which calls the program/script for this SPOR step.
    execution_command: str
    #: Command line options to add to the execution command. These do not
    #: include those from the communication interface these will be, if
    #: applicable, added separately.
    command_options: List[str] = Field(default_factory=list)

    @validator("execution_command")
    @classmethod
    def validate_execution_command_path(cls, v: str):
        """Validator execution_command.

        Check if path to execution command exist.

        Args:
            v (str): Object to validate

        Raises:
            ParserException: If path is not correct, this error is thrown.

        Returns:
            str: original path if no validation error occurs.
        """
        path = pathlib.Path(v)
        if path.exists():
            path = path.expanduser().resolve().as_posix()
        else:
            path = v
        if shutil.which(path) is None:
            raise ParserException(
                "SPORObjectCommandline",
                "unknown",
                f"The execution_command path {v} is not a valid executable.",
            )
        return path

    def run(
        self,
        step_information: StepReturnType,
        environment_id: UUID4,
        validation_id: Optional[int] = None,
        core_count: int = 1,
        reset: bool = False,
    ) -> StepReturnType:
        """This function runs the defined command line command.

        With the defined arguments adding if necessary multi-processing flags
        and the spor communication interface.

        Args:
            step_information (StepReturnType):
                Previously collected step values. Should be up-to-date.
            environment_id (UUID4):
                Environment ID which is used to distinguish different
                environments which run in parallel.
            validation_id (Optional[int], optional):
                During validation the validation id signifies the validation
                episode. If none the current episode is not a validation
                episode. Defaults to None.
            core_count (int, optional):
                Wanted core_count of the task to run.
                Defaults to 1.
            reset (bool, optional):
                Boolean on whether or not this object is called because of a
                reset. Defaults to False.

        Returns:
            StepReturnType:
                The full compliment of the step information are returned.
                The include observation, reward, done, info.
        """
        if self._first_time_setup_not_done:
            self.get_logger().info(
                f"Setup is performed for step with name {self.name}. For "
                f"environment with id {environment_id}."
            )
            self.setup_working_directory(environment_id)
            self._first_time_setup_not_done = False
        # first set up done

        env_logger = self.get_logger()
        step_return: Dict = {
            "observation": step_information[0],
            "reward": step_information[1],
            "done": step_information[2],
            "info": step_information[3],
        }
        # skips the step without doing anything if reset and not run_on_reset
        # add observations if the skipped step has some
        if reset and not self.run_on_reset:
            if self.additional_observations is not None:
                step_return["observation"] = self.get_default_observation(
                    step_return["observation"]
                )
        else:  # executes the step
            multi_proc_prefix = self.get_multiprocessing_prefix(
                core_count=core_count
            )
            command_list = [
                multi_proc_prefix,
                self.execution_command,
                *self.command_options,
                *self.spor_com_interface(
                    reset, environment_id, validation_id, step_information
                ),
            ]

            command = " ".join(command_list)

            # using command line to call the defined command
            exit_code, output = call_commandline(
                command, self.working_directory, env_logger
            )

            step_return["info"][self.name] = {
                "output": output,
                "exit_code": int(exit_code),
            }
            self.get_logger().debug(
                f"The return step return of the current "
                f"command is: {step_return}"
            )
            if exit_code != 0:  # error thrown during command execution
                env_logger.info("SPOR Step thrown error.")
                if self.stop_after_error:
                    env_logger.info(
                        "Will stop episode now, due to thrown error."
                    )
                    step_return["done"] = True
                    # exit code for xns if mesh is tangled hopefully
                    if (
                        self.name == "main_solver" and int(exit_code) == 137
                    ):  # pragma: no cover # this is a XNS specific case
                        step_return["info"]["reset_reason"] = (
                            f"meshTangled-{self.name}"
                        )
                    # the main_solver should always have an output only time
                    # no output is generated srun aborted early (hopefully)
                    elif (
                        self.name == "main_solver" and not output
                    ):  # pragma: no cover # this is a cluster specific case
                        step_return["info"]["reset_reason"] = (
                            f"srunError-{self.name}"
                        )
                        self.get_logger().warning(
                            "Could not find any output of failed command "
                            "assuming srun/mpi error. Trying to exit training "
                            "now."
                        )
                    else:
                        step_return["info"]["reset_reason"] = (
                            f"ExecutionFailed-{self.name}"
                        )
                step_return["reward"] = self.reward_on_error
                if self.additional_observations is not None:
                    step_return["observation"] = self.get_default_observation(
                        step_return["observation"]
                    )
            else:
                if self.use_communication_interface:
                    # if isinstance(output, dict):
                    #     self.spor_com_interface_add(output, step_return)
                    # else:
                    self.spor_com_interface_read(output, step_return)
                if self.reward_on_completion:
                    if step_return["reward"]:
                        env_logger.warning(
                            f"A reward {step_return['reward']} is already set "
                            f"but a default reward {self.reward_on_completion}"
                            " on completion is also set. The default reward "
                            "will overwrite the already set reward. Please "
                            "check if this is the wanted behavior, if not set "
                            "the reward on completion to None."
                        )
                    step_return["reward"] = self.reward_on_completion

        return (
            step_return["observation"],
            step_return["reward"],
            step_return["done"],
            step_return["info"],
        )


SPORObjectTypes = Union[
    SPORObjectCommandLine,
    SPORObjectExternalPythonFunction,
    SPORObjectInternalPythonFunction,
]


class SPORList(BaseModel):
    """The SPORList defines the custom defined steps.

    In the current version these steps take place after the FFD is
    done. These can include the fluid solver, post-processing steps,
    reward generation, additional logging, etc. In a future version the FFD
    might be incorporated into a SPOR step to stream line the internal
    functionality.

    If a step sends a stop signal all consecutive steps will be ignored and the
    Each step can generate observations, rewards and info for the current
    episode step.  These are aggregated as follows.
    """

    #: List of SPOR objects. Each SPOR object is a step  in the environment.
    steps: List[SPORObjectTypes]
    #: Definition on how the reward should be aggregated. Definition for each
    #: option is given :doc:`/SPOR`.
    reward_aggregation: Literal["sum", "min", "max", "mean", "absmax"]

    #: internal data holder for the rewards collected during a single episode.
    #: Could be local variable but is object variable to have to option to
    #: read it out later on, if need be.
    _rewards: List[RewardType] = PrivateAttr(default_factory=list)

    def get_observations(
        self,
    ) -> Optional[
        List[Union[ObservationDefinition, ObservationDefinitionMulti]]
    ]:
        """Aggregate the observations of all steps.

        Returns the aggregated observations as a flattened
        list of observations spaces coming from the SPORSteps.

        Returns:
            int: Number of observations.
        """
        observations = []
        for step in self.steps:
            fi = step.get_observations()
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
        """Aggregate and compute the reward of the SPORList.

        The aggregation method is defined in the variable
        :py:obj:`ReLeSO.spor.SPORList.reward_aggregation`.

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
            elif self.reward_aggregation == "absmax":
                return self._rewards[np.argmax(np.abs(self._rewards))]
            elif self.reward_aggregation == "max":
                return np.max(self._rewards)
            else:
                err_mesg = (
                    f"The given reward aggregation method - "
                    f"{self.reward_aggregation} - is not supported."
                )
                self.get_logger().error(err_mesg)
                raise RuntimeError(err_mesg)
        else:
            self.get_logger().warning(
                "No rewards given. Setting reward to zero (0). Please check if"
                " reward should be given but did not register."
            )
            return 0

    def run(
        self,
        step_information: StepReturnType,
        environment_id: UUID4,
        validation_id: Optional[int] = None,
        core_count: int = 1,
        reset: bool = False,
    ) -> StepReturnType:
        """Run through all steps of the list and handles each output.

        When done all received observations and infos are jointly returned with
        the reward and the indicator whether or not the episode should be
        terminated.

        The values are added in-place to the step_information variable but is
        also returned.

        Args:
            step_information (StepReturnType):
                Previously collected step values. Should be up-to-date.
            environment_id (UUID4):
                Environment ID which is used to distinguish different
                environments which run in parallel.
            validation_id (Optional[int], optional):
                If currently in validation the validation needs to be
                supplied for steps which need this. If none given no validation
                is currently performed. Defaults to None.
            core_count (int, optional):
                Wanted core_count of the task to run. Defaults to 1.
            reset (bool, optional):
                Boolean on whether or not this object is called because of a
                reset. Defaults to False.

        Returns:
            StepReturnType: Results of all steps.
        """
        self._rewards = []
        observations, reward, done, info = step_information
        error_stop = False
        for step in self.steps:
            start = timer()
            if not info.get(step.name):
                info[step.name] = {}
            # error_stop should only be true if releso.SPORObject had a failure
            if error_stop:  # pragma: no cover
                if step.additional_observations:
                    # print("adding zeros")
                    observations = step.get_default_observation(observations)
                    # print(observations, observations.shape)
            else:
                try:
                    observations, reward, done, info = step.run(
                        step_information,
                        environment_id,
                        validation_id,
                        core_count,
                        reset,
                    )
                    if reward is not None:
                        self._rewards.append(reward)
                    # if info:
                    #     join_infos(info, info, self.logger_name)
                except Exception as exp:  # pragma: no cover # noqa: BLE001
                    # This should only be reached if a SPORObject has an error
                    error_message = (
                        f"The current step with name {step.name} has thrown an"
                        f" error: {exp}."
                    )
                    self.get_logger().warning(error_message, exc_info=1)
                    info[step.name]["error_reason"] = error_message
                    if step.stop_after_error:
                        done = True
                        error_stop = True
                        self.get_logger().warning(
                            f"Due to the error in the step {step.name} this"
                            " episode will now be terminated."
                        )
                        if not info.get("reset_reason"):
                            if not (
                                reset_reason := info.get(step.name).get(
                                    "reset_reason"
                                )
                            ):
                                info["reset_reason"] = error_message
                            else:
                                info[step.name]["reset_reason"] = reset_reason
            if done:
                if not info.get("reset_reason"):
                    info["reset_reason"] = "unspecified"
                    self.get_logger().warning(
                        f"Step {step.name} resulted in episode end but did not"
                        " specify reason for termination."
                    )
            reward = self._compute_reward()
            step_information = (observations, reward, done, info)
            end = timer()
            self.get_logger().debug(
                f"SPOR Step {step.name} took {end - start} seconds."
            )
        return step_information
