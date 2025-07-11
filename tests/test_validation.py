import pytest
from pydantic import ValidationError
from stable_baselines3 import PPO

from releso.validation import Validation
from releso.verbosity import VerbosityLevel


@pytest.mark.parametrize(
    [
        "validation_freq",
        "validation_values",
        "save_best_agent",
        "validate_on_training_end",
        "max_timesteps_in_episode",
        "end_episode_on_geometry_not_changed",
        "reward_on_geometry_not_changed",
        "reward_on_episode_exceeds_max_timesteps",
        "val_error",
        "val_warning",
    ],
    [
        (
            1,
            [],
            False,
            False,
            None,
            None,
            None,
            None,
            "at least 1 item",
            False,
        ),
        (1, [12], False, False, None, None, None, None, False, False),
        (
            1,
            [12],
            False,
            False,
            None,
            True,
            None,
            None,
            False,
            "Please set a reward value for geometry not changed",
        ),
        (
            1,
            [12],
            False,
            False,
            5,
            None,
            None,
            None,
            False,
            "Please set a reward value for max time steps exceeded",
        ),
        (
            1,
            [12],
            False,
            False,
            None,
            None,
            -1,
            None,
            "Reward can only be set if end_episode_on_geometry_not_changed",
            False,
        ),
        (
            1,
            [12],
            False,
            False,
            None,
            None,
            None,
            -1,
            "Reward can only be set if max_timesteps_in_episode is a",
            False,
        ),
    ],
)
def test(
    validation_freq,
    validation_values,
    save_best_agent,
    validate_on_training_end,
    max_timesteps_in_episode,
    end_episode_on_geometry_not_changed,
    reward_on_geometry_not_changed,
    reward_on_episode_exceeds_max_timesteps,
    val_error,
    val_warning,
    dir_save_location,
    caplog,
):
    calling_dict = {
        "validation_freq": validation_freq,
        "validation_values": validation_values,
        "save_best_agent": save_best_agent,
        "validate_on_training_end": validate_on_training_end,
        "save_location": dir_save_location,
    }
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
    if reward_on_episode_exceeds_max_timesteps is not None:
        calling_dict["reward_on_episode_exceeds_max_timesteps"] = (
            reward_on_episode_exceeds_max_timesteps
        )

    if val_error:
        with pytest.raises(ValidationError) as err:
            Validation(**calling_dict)
        assert val_error in str(err.value)
        return
    with caplog.at_level(VerbosityLevel.WARNING):
        validation = Validation(**calling_dict)
    if val_warning:
        assert val_warning in caplog.text
    if end_episode_on_geometry_not_changed is None:
        end_episode_on_geometry_not_changed = False
    if (
        max_timesteps_in_episode is not None
        and reward_on_episode_exceeds_max_timesteps is None
    ):
        reward_on_episode_exceeds_max_timesteps = 0
    if (
        end_episode_on_geometry_not_changed
        and reward_on_geometry_not_changed is None
    ):
        reward_on_geometry_not_changed = 0

    assert validation.should_add_callback()
    validation.validation_freq = -1
    assert not validation.should_add_callback()

    val_environment_parameters = (
        validation.get_environment_validation_parameters()
    )
    assert (
        val_environment_parameters["max_timesteps_in_episode"]
        == max_timesteps_in_episode
    )
    assert (
        val_environment_parameters["end_episode_on_geometry_not_changed"]
        == end_episode_on_geometry_not_changed
    )
    assert (
        val_environment_parameters["reward_on_geometry_not_changed"]
        == reward_on_geometry_not_changed
    )
    assert (
        val_environment_parameters["reward_on_episode_exceeds_max_timesteps"]
        == reward_on_episode_exceeds_max_timesteps
    )
    assert val_environment_parameters["validation_values"] == validation_values


@pytest.mark.parametrize("save_location", [(False), (True)])
def test_validation_with_environment(
    save_location, dir_save_location, provide_dummy_environment
):
    calling_dict = {
        "validation_freq": 1,
        "validation_values": [12],
        "save_best_agent": save_location,
        "validate_on_training_end": False,
        "save_location": dir_save_location,
    }
    validation = Validation(**calling_dict)
    assert validation.should_add_callback()
    variable_dict = {
        "eval_environment": provide_dummy_environment,
    }
    if save_location:
        variable_dict["save_location"] = dir_save_location
    call_back = validation.get_callback(**variable_dict)
    assert call_back.eval_freq == 1
    if save_location:
        assert str(dir_save_location) in str(call_back.best_model_save_path)
    else:
        assert call_back.best_model_save_path is None

    agent = PPO("MlpPolicy", provide_dummy_environment, verbose=0, n_steps=100)

    validation.end_validation(agent, provide_dummy_environment)
