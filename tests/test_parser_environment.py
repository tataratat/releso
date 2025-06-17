import copy
import pathlib

import gymnasium as gym
import numpy as np
import pytest
from conftest import dir_save_location_path
from pydantic import ValidationError

from releso.parser_environment import Environment, MultiProcessing
from releso.verbosity import VerbosityLevel


@pytest.mark.parametrize(
    ["n_cores", "w_cores", "error_n_cores"],
    [(1, 1, False), (False, 1, False), (12, 12, False), (-10, False, True)],
)
def test_parser_environment_multiprocessing(
    n_cores, w_cores, error_n_cores, dir_save_location
):
    calling_dict = {
        "save_location": dir_save_location,
    }
    if n_cores:
        calling_dict["number_of_cores"] = n_cores
    if error_n_cores:
        with pytest.raises(ValidationError):
            MultiProcessing(**calling_dict)
        return

    multi = MultiProcessing(**calling_dict)

    assert multi.number_of_cores == w_cores


def test_parser_environment_environment(dir_save_location, default_shape):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)
    env.close()


@pytest.mark.parametrize(
    [
        "n_cores",
        "w_cores",
        "max_timesteps_in_episode",
        "end_episode_on_geometry_not_changed",
        "reward_on_geometry_not_changed",
        "reward_on_episodes_exceeds_max_timesteps",
        "error",
        "warning",
    ],
    [
        (None, 1, None, None, None, None, False, False),
        (
            None,
            1,
            None,
            None,
            None,
            -1,
            "Reward can only be set if max_timesteps_in_episode a positive",
            False,
        ),
        # (
        #     None,
        #     1,
        #     False,
        #     None,
        #     None,
        #     -1,
        #     "Reward can only be set if max_timesteps_in_episode a positive",
        #     False,
        # ), # Somewhat strange behavior, therefore not used as test
        (
            None,
            1,
            10,
            None,
            None,
            None,
            False,
            "Please set a reward value for max time steps exceeded, if",
        ),
        (12, 12, 10, None, None, 0.1, False, False),
        (
            2,
            2,
            None,
            True,
            None,
            None,
            False,
            "Please set a reward value for geometry not changed",
        ),
        (
            None,
            None,
            None,
            None,
            1,
            None,
            "Reward can only be set if end_episode_on_geometry_not_changed ",
            False,
        ),
        (
            None,
            None,
            None,
            False,
            1,
            None,
            "Reward can only be set if end_episode_on_geometry_not_changed ",
            False,
        ),
    ],
)
def test_parser_environment_test_parsing(
    n_cores,
    w_cores,
    max_timesteps_in_episode,
    end_episode_on_geometry_not_changed,
    reward_on_geometry_not_changed,
    reward_on_episodes_exceeds_max_timesteps,
    error,
    warning,
    dir_save_location,
    caplog,
    default_shape,
):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    if n_cores is not None:
        calling_dict["multi_processing"]["number_of_cores"] = n_cores
    if max_timesteps_in_episode is not None:
        calling_dict["max_timesteps_in_episode"] = max_timesteps_in_episode
    if end_episode_on_geometry_not_changed is not None:
        calling_dict["end_episode_on_geometry_not_changed"] = (
            end_episode_on_geometry_not_changed
        )
    if reward_on_geometry_not_changed is not None:
        calling_dict["reward_on_geometry_not_changed"] = (
            reward_on_geometry_not_changed
        )
    if reward_on_episodes_exceeds_max_timesteps is not None:
        calling_dict["reward_on_episode_exceeds_max_timesteps"] = (
            reward_on_episodes_exceeds_max_timesteps
        )
    if error:
        with pytest.raises(ValidationError) as err:
            Environment(**calling_dict)
        assert error in str(err.value)
        return
    with caplog.at_level("WARNING"):
        env = Environment(**calling_dict)
    if warning:
        assert warning in caplog.text
    assert env.is_multiprocessing() == w_cores


def test_parser_environment_observations_no_observation(
    dir_save_location, default_shape
):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": False,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)
    with pytest.raises(RuntimeError) as err:
        env._define_observation_space()
    assert "The observation space is empty." in str(err.value)


def test_parser_environment_observations_single(
    dir_save_location, default_shape
):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": True,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)
    obs_space = env._define_observation_space()
    assert obs_space.shape == (2,)
    observations = {}
    observations["geometry_observation"] = env.geometry.get_observation()
    for step in env.spor.steps:
        observations.update(step.get_default_observation(observations))
    new_obs = np.array(env.check_observations(observations))
    assert new_obs.shape == (2,)


def test_parser_environment_observations_compress(
    dir_save_location, default_shape
):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": True,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                    "additional_observations": [
                        {"name": "test1", "value_min": 0.0, "value_max": 1.0},
                        {"name": "test2", "value_min": 0.0, "value_max": 1.0},
                        {
                            "name": "test3",
                            "value_min": 0.0,
                            "value_max": 1.0,
                            "observation_shape": [
                                9,
                            ],
                            "value_type": "float",
                        },
                    ],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)
    obs_space = env._define_observation_space()
    assert env._flatten_observations
    assert obs_space.shape == (13,)
    observations = {}
    observations["geometry_observation"] = env.geometry.get_observation()
    for step in env.spor.steps:
        observations.update(step.get_default_observation(observations))
    new_obs = np.array(env.check_observations(observations))
    assert new_obs.shape == (13,)


def test_parser_environment_observations_cnn(dir_save_location, default_shape):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": True,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "name": "internal",
                    "save_location": dir_save_location,
                    "reward_on_error": 0.0,
                    "use_communication_interface": True,
                    "working_directory": str(dir_save_location),
                    "function_name": "xns_cnn",
                },
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)
    obs_space = env._define_observation_space()
    assert not env._flatten_observations
    n_envs = 0
    for k in obs_space.spaces.keys():
        assert k in ["cnn_observation", "geometry_observation"]
        n_envs += 1
    assert n_envs == 2
    observations = {}
    observations["geometry_observation"] = env.geometry.get_observation()
    for step in env.spor.steps:
        observations.update(step.get_default_observation(observations))
    new_obs = env.check_observations(observations)
    assert isinstance(new_obs, dict)


def test_parser_environment_observations_non_compressible(
    dir_save_location, default_shape
):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": True,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                    "additional_observations": [
                        {"name": "test1", "value_min": 0.0, "value_max": 1.0},
                        {"name": "test2", "value_min": 0.0, "value_max": 1.0},
                        {
                            "name": "test3",
                            "value_min": 0.0,
                            "value_max": 1.0,
                            "observation_shape": [
                                9,
                                2,
                            ],
                            "value_type": "float",
                        },
                    ],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)
    obs_space = env._define_observation_space()
    assert not env._flatten_observations
    n_envs = 0
    for k in obs_space.spaces.keys():
        assert k in ["test1", "test2", "test3", "geometry_observation"]
        n_envs += 1
    assert n_envs == 4


def test_parser_environment_observations_step_reset_simple(
    dir_save_location, clean_up_provider, caplog, default_shape
):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": True,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)

    gym_env = env.get_gym_environment({
        "logger_name": "test",
        "log_file_location": dir_save_location,
        "logging_level": VerbosityLevel.INFO,
    })
    # clean_up_provider(dir_save_location)
    assert issubclass(type(gym_env), gym.Env)

    obs, info = gym_env.reset()
    assert type(obs) in [list, np.ndarray]
    assert isinstance(info, dict)
    assert len(obs) == 2

    obs, _, _, _, info = gym_env.step(1)
    assert pytest.approx(obs) == [4.9, 5.0]

    with caplog.at_level(VerbosityLevel.WARNING):
        _ = env.get_gym_environment()

    clean_up_provider(dir_save_location)


def test_parser_environment_max_timesteps(
    dir_save_location, clean_up_provider, default_shape
):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": True,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
        "max_timesteps_in_episode": 5,
        "reward_on_episode_exceeds_max_timesteps": -1,
    }
    env = Environment(**calling_dict)

    gym_env = env.get_gym_environment()
    gym_env.reset()
    for idx in range(4):
        obs, _, done, _, info = gym_env.step(1)
        assert pytest.approx(obs) == [5 - ((idx + 1) * 0.1), 5.0]
        assert not done
    obs, _, done, _, info = gym_env.step(1)
    assert pytest.approx(obs) == [4.5, 5.0]
    assert done
    assert info["reset_reason"] == "max_timesteps_exceeded"

    clean_up_provider(dir_save_location)


def test_parser_environment_geometry_not_changed(
    dir_save_location, clean_up_provider, default_shape
):
    dir_save_location_path(True)
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": True,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "execution_command": "echo",
                    "command_options": ["'test'"],
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
        "end_episode_on_geometry_not_changed": True,
        "reward_on_geometry_not_changed": -1,
    }
    env = Environment(**calling_dict)

    gym_env = env.get_gym_environment()
    gym_env.reset()
    for idx in range(4):
        obs, _, done, _, info = gym_env.step(1)
        assert pytest.approx(obs) == [5 - ((idx + 1) * 0.1), 5.0]
        assert not done
    # this should increase the second variable which is at its bound so it will
    #  not change
    obs, _, done, _, info = gym_env.step(2)
    assert pytest.approx(obs) == [4.6, 5.0]
    assert done
    assert info["reset_reason"] == "geometry_not_changed"

    clean_up_provider(dir_save_location)


def test_parser_environment_non_geometry_observations(
    dir_save_location, clean_up_provider, default_shape
):
    dir_save_location_path(True)
    file_path = pathlib.Path(__file__).parent
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": False,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "python_file_path": (
                        f"{str(file_path)}/samples"
                        "/spor_python_scripts_tests/file_exists_has_main.py"
                    ),
                    "use_communication_interface": True,
                    "additional_observations": 3,
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)

    gym_env = env.get_gym_environment()
    gym_env.reset()
    obs, _, _, _, _ = gym_env.step(2)
    assert obs == pytest.approx([1, 2, 3])
    clean_up_provider(dir_save_location)


def test_parser_environment_validation(
    dir_save_location, clean_up_provider, default_shape
):
    dir_save_location_path(True)
    file_path = pathlib.Path(__file__).parent
    calling_dict = {
        "save_location": dir_save_location,
        "geometry": {
            "save_location": dir_save_location,
            "shape_definition": default_shape,
            "action_based_observation": False,
        },
        "spor": {
            "save_location": dir_save_location,
            "steps": [
                {
                    "save_location": dir_save_location,
                    "name": "test",
                    "reward_on_error": -1,
                    "working_directory": str(dir_save_location),
                    "python_file_path": (
                        f"{str(file_path)}/samples"
                        "/spor_python_scripts_tests/file_exists_has_main.py"
                    ),
                    "use_communication_interface": True,
                    "additional_observations": 3,
                }
            ],
            "reward_aggregation": "sum",
        },
        "multi_processing": {
            "save_location": dir_save_location,
        },
    }
    env = Environment(**calling_dict)
    val_values = [1, 2, 3]
    env.set_validation(
        validation_values=copy.deepcopy(val_values),
    )
    gym_env = env.get_gym_environment()
    gym_env.reset()
    for _ in range(len(val_values)):
        _ = gym_env.step(2)
        assert (val_v := env.get_validation_id()) in val_values
        gym_env.reset()
        val_values.remove(val_v)
    assert len(val_values) == 0
    clean_up_provider(dir_save_location)
