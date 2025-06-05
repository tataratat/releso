import copy
import pathlib
import shutil
import uuid
from collections.abc import Iterable
from typing import List

import numpy as np
import pytest
from conftest import dir_save_location_path
from pydantic import ValidationError

from releso.observation import (
    ObservationDefinition,
    ObservationDefinitionMulti,
)
from releso.spor import (  # SPORObjectPythonFunction,; SPORObjectTypes,
    MPIClusterMultiProcessor,
    MultiProcessor,
    SPORList,
    SPORObject,
    SPORObjectCommandLine,
    SPORObjectExecutor,
    SPORObjectExternalPythonFunction,
    SPORObjectInternalPythonFunction,
)
from releso.verbosity import VerbosityLevel


def get_obs_name(object_to_check) -> List[str]:
    if isinstance(
        object_to_check, (ObservationDefinition, ObservationDefinitionMulti)
    ):
        return [object_to_check.name]
    elif isinstance(object_to_check, Iterable):
        return [elem.name for elem in object_to_check]
    elif object_to_check is None:
        return []
    else:
        raise RuntimeError(f"{type(object_to_check)}, object_to_check")


@pytest.mark.parametrize(
    [
        "command",
        "max_core_count",
        "external_core_count",
        "always_use",
        "resulting_count",
    ],
    [
        (None, 1, 1, None, 1),
        (None, 10, 1, None, 1),
        (None, 10, 10, None, 10),
        (None, 1, 10, False, 1),
        (None, 10, -5, False, 1),
        (None, 1, 1, True, 1),
        (None, 10, 1, True, 1),
        (None, 10, 10, True, 10),
        (None, 1, 10, True, 1),
        (None, 10, -5, True, 1),
        ("this_is_a_command", 2, 2, True, 2),
    ],
)
def test_multi_processor(
    command,
    max_core_count,
    external_core_count,
    always_use,
    resulting_count,
    dir_save_location,
):
    calling_dict = {
        "save_location": dir_save_location,
        "max_core_count": max_core_count,
    }
    if command is not None:
        calling_dict["command"] = command
    if always_use is not None:
        calling_dict["always_use"] = always_use
    mp = MultiProcessor(**calling_dict)

    ret_command = mp.get_command(external_core_count)

    if resulting_count == 1 and not always_use:
        assert ret_command == ""
    else:
        if command is None:
            assert ret_command == f"mpiexec -n {resulting_count}"
        else:
            assert ret_command == f"{command} {resulting_count}"


@pytest.mark.parametrize(
    [
        "command",
        "command_variable",
        "mpi_flags_variable",
        "mpi_flags_variable_variable",
        "max_core_count",
        "external_core_count",
        "always_use",
        "resulting_command",
        "error",
    ],
    [
        (None, "mpiexec", None, "-np 2", 1, 1, True, "$MPIEXEC -np 1", False),
        (None, "mpiexec", None, "-np 2", 1, 1, False, "", False),
        (
            None,
            "total_annihilation",
            None,
            "",
            10,
            10,
            None,
            "$MPIEXEC -n 10",
            False,
        ),
        (
            "$ANOTHER_COMMAND",
            "simple_times",
            None,
            "los 3",
            5,
            -1,
            True,
            "$ANOTHER_COMMAND los 1",
            False,
        ),
        (
            "$ANOTHER_COMMAND",
            "simple_times",
            None,
            "los as 3",
            5,
            -1,
            True,
            "$ANOTHER_COMMAND los 1",
            "MPI cluster environment variable not as expected.",
        ),
        # (None, "mpiexec", 10, 10, None, 10),
        # (None, "mpiexec", 10, -5, False, 1),
        # (None, "mpiexec", 10, 1, True, 1),
        # (None, "mpiexec", 10, -5, True, 1),
        # ("$this_is_a_command", 2, 2, True, 2),
    ],
)
def test_mpi_cluster_multi_processor(
    command,
    command_variable,
    mpi_flags_variable,
    mpi_flags_variable_variable,
    max_core_count,
    external_core_count,
    always_use,
    resulting_command,
    error,
    dir_save_location,
    monkeypatch,
):
    calling_dict = {
        "save_location": dir_save_location,
        "location": "cluster",
        "max_core_count": max_core_count,
    }
    if always_use is not None:
        calling_dict["always_use"] = always_use
    if command is not None:
        calling_dict["command"] = command
    if command_variable is not None:
        monkeypatch.setenv(
            command[1:] if command else "MPIEXEC", command_variable
        )
    if mpi_flags_variable is not None:
        calling_dict["mpi_flags_variable"] = mpi_flags_variable
    if mpi_flags_variable_variable is not None:
        monkeypatch.setenv(
            mpi_flags_variable[1:]
            if mpi_flags_variable
            else "FLAGS_MPI_BATCH",
            mpi_flags_variable_variable,
        )
    mp = MPIClusterMultiProcessor(**calling_dict)
    mp.get_logger().handlers.clear()
    if error:
        with pytest.raises(RuntimeError):
            ret_command = mp.get_command(external_core_count)
        return
    ret_command = mp.get_command(external_core_count)
    assert ret_command == resulting_command


@pytest.mark.parametrize(
    "additional_observations, obs_type",
    [
        (12, ObservationDefinition),
        (str(0), None),
        (
            {"name": "test", "value_min": 0.0, "value_max": 1.0},
            ObservationDefinition,
        ),
        (
            [
                {"name": "test", "value_min": 0.0, "value_max": 1.0},
                ({"name": "test", "value_min": 0.0, "value_max": 1.0}),
            ],
            (list, ObservationDefinition),
        ),
    ],
)
def test_spor_object(
    additional_observations, obs_type, dir_save_location, caplog
):
    # get_parser_logger().handlers.clear()
    calling_dict = {
        "name": "test",
        "save_location": dir_save_location,
        "reward_on_error": 0.0,
    }
    if additional_observations:
        calling_dict["additional_observations"] = additional_observations
    with caplog.at_level(VerbosityLevel.WARNING):
        spor_object = SPORObject(**calling_dict)
    # We might want to deprecate the hidden option to instantiate additional
    # observations via an int. Building up a default observation definition.
    # if isinstance(additional_observations, int):
    #     assert (
    #         f"Please do not use this method to register additional observati"
    #         in caplog.text
    #     )
    obs = spor_object.get_observations()
    # check if observations are of correct type
    if isinstance(obs_type, tuple):
        # only for 2 element tuple like (list, ObservationDefinition)
        # only checks the first element of the obs if it is iterable
        element_to_check = obs
        for obs_type_item in obs_type:
            if obs_type_item is list:
                assert isinstance(element_to_check, obs_type_item)
                element_to_check = element_to_check[0]
            else:
                assert isinstance(element_to_check, obs_type_item)
    elif obs_type is None:
        assert obs is None
    else:
        assert issubclass(obs.__class__, obs_type)
    existing_obs = {}
    existing_obs_copy = copy.deepcopy(existing_obs)
    default_obs = spor_object.get_default_observation(existing_obs)
    if obs_type is None:
        assert default_obs == existing_obs
    else:
        assert default_obs != existing_obs_copy


@pytest.mark.parametrize(
    [
        "multi_processor",
        "wanted_mp",
        "use_communication_interface",
        "add_step_information",
        "working_dir",
        "error",
    ],
    [
        (
            {"max_core_count": 12},
            MultiProcessor,
            None,
            None,
            False,
            "does not exist or is not a valid directory",
        ),
        (
            {"max_core_count": 12},
            MultiProcessor,
            None,
            None,
            True,
            False,
        ),
        (
            {"max_core_count": 12},
            MultiProcessor,
            None,
            True,
            True,
            "Please only set the add_step_information variable to True",
        ),
        ({"max_core_count": 12}, MultiProcessor, True, True, True, False),
        (
            {"max_core_count": 12, "location": "cluster"},
            MPIClusterMultiProcessor,
            True,
            True,
            True,
            False,
        ),
    ],
)
def test_spor_object_executor(
    multi_processor,
    wanted_mp,
    use_communication_interface,
    add_step_information,
    working_dir,
    error,
    dir_save_location,
):
    calling_dict = {
        "name": "test",
        "save_location": dir_save_location,
        "reward_on_error": 0.0,
    }
    if working_dir:
        calling_dict["working_directory"] = str(dir_save_location_path(True))
    else:
        calling_dict["working_directory"] = str(dir_save_location_path())
    if multi_processor:
        multi_processor["save_location"] = dir_save_location
        calling_dict["multi_processor"] = multi_processor
    if use_communication_interface:
        calling_dict["use_communication_interface"] = (
            use_communication_interface
        )
    if add_step_information:
        calling_dict["add_step_information"] = add_step_information
    if error:
        with pytest.raises(ValidationError) as err:
            spor_object = SPORObjectExecutor(**calling_dict)
        assert error in str(err.value)
        return
    spor_object = SPORObjectExecutor(**calling_dict)
    assert type(spor_object.multi_processor) is wanted_mp


@pytest.mark.parametrize(
    "working_dir, wanted_result, relative, pre_create, error",
    [
        (".", ".", True, False, False),
        (
            "{}/test",
            "test_save_location_please_delete/test/",
            False,
            False,
            False,
        ),
        ("non_existing/folder", "as/", False, False, True),
        ("existing/folder/", "existing/folder/", True, True, False),
        (
            "{}/test/{}",
            "test_save_location_please_delete/test/idididid/",
            False,
            False,
            False,
        ),
    ],
)
def test_spor_object_executor_working_dir(
    working_dir, wanted_result, relative, pre_create, error, dir_save_location
):
    if pre_create:
        pathlib.Path(working_dir).mkdir(parents=True, exist_ok=True)

    if not relative:
        wanted_result = str(
            (
                pathlib.Path("/".join(str(dir_save_location).split("/")[:-1]))
                / wanted_result
            )
            .resolve()
            .absolute()
        )

    calling_dict = {
        "name": "test",
        "save_location": dir_save_location,
        "reward_on_error": 0.0,
        "multi_processor": {
            "max_core_count": 1,
            "save_location": dir_save_location,
        },
    }
    calling_dict["working_directory"] = str(working_dir)
    if error:
        with pytest.raises(ValidationError):
            spor_object = SPORObjectExecutor(**calling_dict)
            spor_object.setup_working_directory("idididid")
        return
    spor_object = SPORObjectExecutor(**calling_dict)
    spor_object.setup_working_directory("idididid")
    print(spor_object.working_directory)
    # if relative:
    #     assert spor_object.working_directory == wanted_result
    # else:
    assert spor_object.working_directory == wanted_result
    if pre_create:
        shutil.rmtree(wanted_result)


@pytest.mark.parametrize("with_multi_processing", [(True), (False)])
def test_spor_object_executor_setup_working_dir(
    with_multi_processing, dir_save_location, caplog, clean_up_provider
):
    dir_save_location_path(True)
    calling_dict = {
        "name": "test",
        "save_location": dir_save_location,
        "reward_on_error": 0.0,
        "multi_processor": {
            "max_core_count": 12,
            "save_location": dir_save_location,
        },
    }
    if with_multi_processing:
        calling_dict["working_directory"] = str(dir_save_location) + "/{}"
    else:
        calling_dict["working_directory"] = str(dir_save_location)

    spor_object = SPORObjectExecutor(**calling_dict)
    env_id = uuid.uuid4()
    with caplog.at_level(VerbosityLevel.INFO):
        spor_object.setup_working_directory(env_id)
    if with_multi_processing:
        assert "Found placeholder for step with name" in caplog.text
        assert (dir_save_location / str(env_id)).is_dir()
    else:
        assert "Found placeholder for step with name" not in caplog.text
        assert not (dir_save_location / str(env_id)).is_dir()
    clean_up_provider(dir_save_location)

    assert "" == spor_object.get_multiprocessing_prefix(1)
    assert "12" in spor_object.get_multiprocessing_prefix(12)


@pytest.mark.parametrize(
    "use_comm_interface, reset, validation_id, step_info, add_step_info",
    [
        (False, False, None, None, False),
        (True, False, None, None, False),
        (True, True, None, None, False),
        (True, False, 1234, None, False),
        (
            True,
            False,
            None,
            ["something", "something else", 1234, 0.1234],
            True,
        ),
        (
            True,
            True,
            1234,
            ["something", "something else", 1234, 0.1234],
            False,
        ),
    ],
)
def test_spor_object_executor_spor_com_interface(
    use_comm_interface,
    reset,
    validation_id,
    step_info,
    add_step_info,
    dir_save_location,
    caplog,
    clean_up_provider,
):
    dir_save_location_path(True)
    calling_dict = {
        "name": "test",
        "save_location": dir_save_location,
        "reward_on_error": 0.0,
        "multi_processor": {
            "max_core_count": 12,
            "save_location": dir_save_location,
        },
        "working_directory": str(dir_save_location),
        "use_communication_interface": use_comm_interface,
        "add_step_information": add_step_info,
    }
    spor_object = SPORObjectExecutor(**calling_dict)
    env_id = uuid.uuid4()
    func_calling_dict = {
        "environment_id": env_id,
        "step_information": step_info,
        "reset": reset,
        "validation_id": validation_id,
    }
    # raise RuntimeError(f"{step_info}")
    interface_string = spor_object.spor_com_interface(**func_calling_dict)
    if not use_comm_interface:
        assert len(interface_string) == 0
        return
    interface_string = " ".join(interface_string)
    if reset:
        assert "--reset" in interface_string
    else:
        assert "--reset" not in interface_string
    if validation_id is not None:
        assert f"--validation_value {str(validation_id)}" in interface_string
    else:
        assert "--validation_value" not in interface_string
    if add_step_info:
        assert "--json_object" in interface_string
    else:
        assert "--json_object" not in interface_string


@pytest.mark.parametrize(
    "observation, observation_addition, old_done, new_done, reset_reason",
    [
        (3, [1, 2, 3], False, False, None),
        (3, [1, 2, 3], False, False, "reset_reason"),
        (
            [
                {"name": "test1", "value_min": 0.0, "value_max": 1.0},
                {"name": "test2", "value_min": 0.0, "value_max": 1.0},
            ],
            {"test1": 0.5, "test2": 0.5},
            False,
            False,
            None,
        ),
        (None, None, False, False, None),
    ],
)
def test_spor_object_executor_interface_add(
    observation,
    observation_addition,
    old_done,
    new_done,
    reset_reason,
    dir_save_location,
):
    dir_save_location_path(True)
    calling_dict = {
        "name": "test",
        "save_location": dir_save_location,
        "reward_on_error": 0.0,
        "multi_processor": {
            "max_core_count": 12,
            "save_location": dir_save_location,
        },
        "working_directory": str(dir_save_location),
    }
    if observation is not None:
        calling_dict["additional_observations"] = observation
    spor_object = SPORObjectExecutor(**calling_dict)
    new_step_info = {
        "observations": observation_addition,
        "info": {spor_object.name: {"ret_info": 42}},
        "done": new_done,
        "reward": 0.0,
    }
    if reset_reason is not None:
        new_step_info["info"]["reset_reason"] = reset_reason
    # raise RuntimeError(f"{new_step_info}")
    step_info = {
        "observation": {},
        "info": {spor_object.name: {}},
        "done": new_done,
        "reward": -10.0,
    }

    spor_object.spor_com_interface_add(
        returned_step_dict=new_step_info, step_dict=step_info
    )
    if reset_reason is not None:
        assert reset_reason == step_info["info"]["reset_reason"]
    assert (old_done or new_done) == step_info["done"]
    assert step_info["reward"] == 0.0


@pytest.mark.parametrize(
    [
        "observation",
        "observation_addition",
        "old_done",
        "new_done",
        "reset_reason",
        "error",
    ],
    [
        (3, [1, 2, 3], False, False, None, False),
        (3, [1, 2, 3], False, False, "reset_reason", False),
        (
            [
                {"name": "test1", "value_min": 0.0, "value_max": 1.0},
                {"name": "test2", "value_min": 0.0, "value_max": 1.0},
            ],
            {"test1": 0.5, "test2": 0.5},
            False,
            False,
            None,
            False,
        ),
        (None, None, False, False, None, False),
        (None, None, False, False, None, True),
    ],
)
def test_spor_object_executor_interface_read(
    observation,
    observation_addition,
    old_done,
    new_done,
    reset_reason,
    error,
    dir_save_location,
):
    dir_save_location_path(True)
    calling_dict = {
        "name": "test",
        "save_location": dir_save_location,
        "reward_on_error": 0.0,
        "multi_processor": {
            "max_core_count": 12,
            "save_location": dir_save_location,
        },
        "working_directory": str(dir_save_location),
    }
    if observation is not None:
        calling_dict["additional_observations"] = observation
    spor_object = SPORObjectExecutor(**calling_dict)
    new_step_info = {
        "observations": observation_addition,
        "info": {spor_object.name: {"ret_info": 42}},
        "done": new_done,
        "reward": 0.0,
    }
    if reset_reason is not None:
        new_step_info["info"]["reset_reason"] = reset_reason
    # raise RuntimeError(f"{new_step_info}")
    step_info = {
        "observation": {},
        "info": {spor_object.name: {}},
        "done": new_done,
        "reward": -10.0,
    }
    if error:
        with pytest.raises(SyntaxError):
            spor_object.spor_com_interface_read(
                output=bytes(str(new_step_info)[4:], "utf-8"),
                step_dict=step_info,
            )
        return
    b_ret_dict = bytes(str(new_step_info), "utf-8")
    spor_object.spor_com_interface_read(output=b_ret_dict, step_dict=step_info)
    if reset_reason is not None:
        assert reset_reason == step_info["info"]["reset_reason"]
    assert (old_done or new_done) == step_info["done"]
    assert step_info["reward"] == 0.0
    obs_names = get_obs_name(spor_object.additional_observations)
    if len(obs_names) == 1:
        obs_name = obs_names[0]
        assert np.allclose(
            step_info["observation"][obs_name],
            new_step_info["observations"],
        )
    else:
        for obs_name in obs_names:
            assert np.allclose(
                step_info["observation"][obs_name],
                new_step_info["observations"][obs_name],
            )


@pytest.mark.parametrize(
    [
        "reset",
        "run_on_reset",
        "additional_observations",
        "patch_function",
        "reward_on_completion",
    ],
    [
        (False, False, 3, False, None),
        (True, False, 3, False, None),
        (False, False, None, False, None),
        (True, True, None, False, None),
        (True, True, None, True, None),
        (True, True, None, True, 12),
    ],
)
def test_spor_internal_python_function(
    reset,
    run_on_reset,
    additional_observations,
    patch_function,
    reward_on_completion,
    dir_save_location,
    monkeypatch,
    caplog,
):
    dir_save_location_path(True)
    if patch_function:
        from releso.util import cnn_xns_observations

        def patch_function(*args, **kwargs):
            return {
                "reward": 2,
                "info": {},
                "done": False,
                "observations": np.zeros((3, 200, 200)),
            }, args[2]

        monkeypatch.setattr(cnn_xns_observations, "main", patch_function)

    calling_dict = {
        "name": "xns_cnn",
        "function_name": "xns_cnn",
        "save_location": dir_save_location,
        "working_directory": str(dir_save_location),
        "reward_on_error": 0.0,
        "use_communication_interface": True,
        "run_on_reset": run_on_reset,
    }
    if additional_observations is not None:
        calling_dict["additional_observations"] = additional_observations
    if reward_on_completion is not None:
        calling_dict["reward_on_completion"] = reward_on_completion
    spor_object = SPORObjectInternalPythonFunction(**calling_dict)

    env_id = uuid.uuid4()
    observations = {}
    done = False
    reward = 0.0
    info = {}
    with caplog.at_level(VerbosityLevel.INFO):
        observations, reward, done, info = spor_object.run(
            (observations, reward, done, info),
            environment_id=env_id,
            reset=reset,
        )
    if reward_on_completion is not None:
        assert reward == reward_on_completion
        assert (
            "The default reward will overwrite the already set reward"
            in caplog.text
        )


@pytest.mark.parametrize(
    "file_path, error_init, error_run",
    [
        (
            "/samples/spor_python_scripts_tests/file_exists_has_main.py",
            False,
            False,
        ),
        (
            "/samples/spor_python_scripts_tests/file_exists_no_main.py",
            False,
            "Could not get main function from python file",
        ),
        (
            "/samples/spor_python_scripts_tests/file_exists_unknown_import.py",
            False,
            "Could not load the python file",
        ),
        (
            "/samples/spor_python_scripts_tests/file_exists.py",
            "does not exist.",
            False,
        ),
    ],
)
def test_spor_external_python_function(
    file_path, error_init, error_run, dir_save_location, clean_up_provider
):
    dir_save_location_path(True)
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    calling_dict = {
        "name": "dummy_spor",
        "python_file_path": file_prefix + file_path,
        "save_location": dir_save_location,
        "working_directory": str(dir_save_location),
        "reward_on_error": 0.0,
        "use_communication_interface": True,
        "additional_observations": 3,
    }
    if error_init:
        with pytest.raises(ValidationError) as err:
            spor_obj = SPORObjectExternalPythonFunction(**calling_dict)
        assert error_init in str(err.value)

        clean_up_provider(dir_save_location)
        return
    spor_obj = SPORObjectExternalPythonFunction(**calling_dict)

    env_id = uuid.uuid4()
    observations = {}
    done = False
    reward = 0.0
    info = {}
    if error_run:
        with pytest.raises(RuntimeError) as err:
            spor_obj.run(
                (observations, reward, done, info), environment_id=env_id
            )
        assert error_run in str(err.value)
        clean_up_provider(dir_save_location)
        return
    observations, reward, done, info = spor_obj.run(
        (observations, reward, done, info), environment_id=env_id
    )
    assert reward == 2
    assert np.allclose(
        observations[spor_obj.additional_observations.name], [1, 2, 3]
    )
    clean_up_provider(dir_save_location)


def test_spor_external_python_function_reset_reason(dir_save_location):
    dir_save_location_path(True)
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    calling_dict = {
        "name": "dummy_spor",
        "python_file_path": (
            file_prefix
            + "/samples/spor_python_scripts_tests/file_exists_has_main.py"
        ),
        "save_location": dir_save_location,
        "working_directory": str(dir_save_location),
        "reward_on_error": 0.0,
        "use_communication_interface": True,
        "additional_observations": 3,
    }
    spor_obj = SPORObjectExternalPythonFunction(**calling_dict)

    env_id = uuid.uuid4()
    observations = {}
    done = False
    reward = 0.0
    info = {}
    while not done:
        observations, reward, done, info = spor_obj.run(
            (observations, reward, done, info), environment_id=env_id
        )
    assert done
    assert info[spor_obj.name]["reset_reason"] == "func_data > 10"
    assert info["reset_reason"] == "func_data > 10"


@pytest.mark.parametrize(
    [
        "execution_command",
        "error_init",
        "command_options",
        "error_run",
        "reward_on_completion",
        "reset",
    ],
    [
        (
            "python",
            None,
            "samples/spor_python_scripts_tests/file_exists_command_line.py",
            False,
            False,
            False,
        ),
        (
            "python",
            None,
            "samples/spor_python_scripts_tests/file_exists_command_line.py",
            False,
            True,
            False,
        ),
        (
            "python",
            None,
            "samples/spor_python_scripts_tests/file_exists_command_line.py",
            False,
            False,
            True,
        ),
        (
            "python",
            None,
            "samples/spor_python_scripts_tests/file_exists_command_l.py",
            "Some error",
            False,
            False,
        ),
        (
            str(dir_save_location_path(True)),
            "The execution_command path",
            False,
            False,
            False,
            False,
        ),
    ],
)
def test_spor_command_line(
    execution_command,
    error_init,
    command_options,
    error_run,
    reward_on_completion,
    reset,
    clean_up_provider,
    dir_save_location,
    caplog,
):
    dir_save_location_path(True)
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    calling_dict = {
        "name": "dummy_spor",
        "execution_command": execution_command,
        "save_location": dir_save_location,
        "working_directory": str(dir_save_location),
        "reward_on_error": 0.0,
        "use_communication_interface": True,
        "additional_observations": 3,
    }
    if command_options:
        calling_dict["command_options"] = [f"{file_prefix}/{command_options}"]
    if reward_on_completion:
        calling_dict["reward_on_completion"] = 12
    if reset:
        calling_dict["run_on_reset"] = False
    if error_init:
        with pytest.raises(ValidationError) as err:
            spor_obj = SPORObjectCommandLine(**calling_dict)
        assert error_init in str(err.value)

        clean_up_provider(dir_save_location)
        return
    spor_obj = SPORObjectCommandLine(**calling_dict)

    # error_run = False
    env_id = uuid.uuid4()
    observations = {}
    done = False
    reward = 0.0
    info = {}
    # create folder for step
    file_path = dir_save_location / f"{env_id}/{spor_obj._run_id}.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    observations, reward, done, info = spor_obj.run(
        (observations, reward, done, info),
        environment_id=env_id,
        reset=reset,
    )
    if error_run and error_run is not None:
        # default observation
        assert np.allclose(
            observations[spor_obj.additional_observations.name], [-1, -1, -1]
        )
    elif reward_on_completion:
        assert reward == 12
        assert np.allclose(
            observations[spor_obj.additional_observations.name], [1, 2, 3]
        )
    elif reset:
        # default observation
        assert np.allclose(
            observations[spor_obj.additional_observations.name], [-1, -1, -1]
        )
        assert pytest.approx(0) == reward
    else:
        assert reward == 2
        assert np.allclose(
            observations[spor_obj.additional_observations.name], [1, 2, 3]
        )
    clean_up_provider(dir_save_location)


@pytest.mark.parametrize(
    [
        "rewards",
        "reward_aggregation",
        "change_aggregator",
        "w_sum",
        "error",
    ],
    [
        ([1, 2, 3, 4.0], "mean", None, 2.5, False),
        ([1, 2, 3, 4.0], "sum", None, 10, False),
        ([1, 2, 3, 4.0], "mean", "sum", 10, False),
        ([1, 2, 3, 4.0], "min", None, 1, False),
        ([-25, 1, 2, 3, 4.0], "max", None, 4, False),
        ([1, 2, 3, 4.0], "absmax", None, 4, False),
        ([-25, 1, 2, 3, 4.0], "absmax", None, -25, False),
        (
            [1, 2, 3, 4.0],
            "mean",
            "test",
            4,
            "The given reward aggregation method",
        ),
        ([], "absmax", None, 0, False),
    ],
)
def test_spor_list_compute_reward(
    dir_save_location,
    rewards,
    reward_aggregation,
    change_aggregator,
    w_sum,
    error,
):
    calling_dict = {
        "save_location": dir_save_location,
        "steps": [],
        "reward_aggregation": reward_aggregation,
    }
    spor_list = SPORList(**calling_dict)
    spor_list._rewards = rewards
    if change_aggregator:
        spor_list.reward_aggregation = change_aggregator
    if error:
        with pytest.raises(RuntimeError) as err:
            spor_list._compute_reward()
        assert error in str(err.value)
        return
    reward = spor_list._compute_reward()
    assert pytest.approx(w_sum) == reward


def test_spor_list_get_observations_correct_spor_type(dir_save_location):
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    calling_dict = {
        "save_location": dir_save_location,
        "steps": [
            {
                "name": "external",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "additional_observations": [
                    {"name": "test1", "value_min": 0.0, "value_max": 1.0},
                    {"name": "test2", "value_min": 0.0, "value_max": 1.0},
                ],
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
            {
                "name": "internal",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "function_name": "xns_cnn",
            },
            {
                "name": "command_line",
                "save_location": dir_save_location,
                "execution_command": "python",
                "command_options": [
                    (
                        f"{file_prefix}/samples/spor_python_scripts_tests/"
                        "file_exists_command_line.py"
                    )
                ],
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
            },
        ],
        "reward_aggregation": "mean",
    }
    dir_save_location_path(True)
    spor_list = SPORList(**calling_dict)
    for idx, spor_obj in enumerate(spor_list.steps):
        if idx == 0:
            assert type(spor_obj) is SPORObjectExternalPythonFunction
            assert type(spor_obj) not in [
                SPORObjectInternalPythonFunction,
                SPORObjectCommandLine,
            ]
        elif idx == 1:
            assert type(spor_obj) is SPORObjectInternalPythonFunction
            assert type(spor_obj) not in [
                SPORObjectExternalPythonFunction,
                SPORObjectCommandLine,
            ]
        elif idx == 2:
            assert type(spor_obj) is SPORObjectCommandLine
            assert type(spor_obj) not in [
                SPORObjectExternalPythonFunction,
                SPORObjectInternalPythonFunction,
            ]
    obs = spor_list.get_observations()
    assert len(obs) == 3


def test_spor_list_observations_none(dir_save_location):
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    dir_save_location_path(True)
    # check for None obs
    calling_dict = {
        "save_location": dir_save_location,
        "steps": [
            {
                "name": "external",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
        ],
        "reward_aggregation": "mean",
    }
    spor_list = SPORList(**calling_dict)
    assert spor_list.get_observations() is None


def test_spor_list_run_simple(dir_save_location, clean_up_provider):
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "steps": [
            {
                "name": "external",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
        ],
        "reward_aggregation": "mean",
    }
    env_id = uuid.uuid4()
    observations = {}
    done = False
    reward = 0.0
    info = {}
    spor_list = SPORList(**calling_dict)
    observations, reward, done, info = spor_list.run(
        (observations, reward, done, info),
        environment_id=env_id,
    )
    clean_up_provider(dir_save_location)


def test_spor_list_run_observations(dir_save_location, clean_up_provider):
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "steps": [
            {
                "name": "external",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "additional_observations": 3,
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
            {
                "name": "external1",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "additional_observations": 3,
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
            {
                "name": "external2",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "additional_observations": 3,
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
            {
                "name": "external3",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "additional_observations": 3,
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
        ],
        "reward_aggregation": "mean",
    }
    env_id = uuid.uuid4()
    observations = {}
    done = False
    reward = 0.0
    info = {}
    spor_list = SPORList(**calling_dict)
    observations, reward, done, info = spor_list.run(
        (observations, reward, done, info),
        environment_id=env_id,
    )
    obs = spor_list.get_observations()
    assert len(observations.keys()) == len(obs) == 4
    for ob in obs:
        obs_name = ob.name
        assert np.allclose(observations[obs_name], [1, 2, 3])

    clean_up_provider(dir_save_location)


def test_spor_list_run_done(dir_save_location, clean_up_provider):
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "steps": [
            {
                "name": "external",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
        ],
        "reward_aggregation": "mean",
    }
    env_id = uuid.uuid4()
    observations = {}
    done = True
    reward = 0.0
    info = {}
    spor_list = SPORList(**calling_dict)
    observations, reward, done, info = spor_list.run(
        (observations, reward, done, info),
        environment_id=env_id,
    )
    assert info.get("reset_reason") == "unspecified"
    clean_up_provider(dir_save_location)


def test_spor_list_run_reset(dir_save_location, clean_up_provider, caplog):
    file_prefix = str(pathlib.Path(__file__).parent.resolve())
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "steps": [
            {
                "name": "produces_error",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "working_directory": str(dir_save_location),
                "execution_command": "python",
                "command_options": [
                    str(dir_save_location / "does_not_exist.txt")
                ],
                "stop_after_error": True,
            },
            {
                "name": "external",
                "save_location": dir_save_location,
                "reward_on_error": 0.0,
                "use_communication_interface": True,
                "working_directory": str(dir_save_location),
                "run_on_reset": False,
                "additional_observations": 3,
                "python_file_path": (
                    f"{file_prefix}/samples/spor_python_scripts_tests/"
                    "file_exists_has_main.py"
                ),
            },
        ],
        "reward_aggregation": "mean",
    }
    env_id = uuid.uuid4()
    observations = {}
    done = False
    reward = 0.0
    info = {}
    spor_list = SPORList(**calling_dict)
    with caplog.at_level(VerbosityLevel.WARNING):
        observations, reward, done, info = spor_list.run(
            (observations, reward, done, info),
            environment_id=env_id,
            reset=True,
        )
    assert done

    clean_up_provider(dir_save_location)
    clean_up_provider(dir_save_location)
