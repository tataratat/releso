from typing import Any, Dict, List, Union
import pathlib

import pytest

from releso.base_parser import BaseParser
from releso.exceptions import ValidationNotSet


def recursive_remove_save_location(
    dict_to_clean: Union[List[Any], Dict[str, Any]],
):
    if isinstance(dict_to_clean, list):
        for item in dict_to_clean:
            recursive_remove_save_location(item)
    if isinstance(dict_to_clean, dict):
        if "save_location" in dict_to_clean:
            del dict_to_clean["save_location"]
        for value in dict_to_clean.values():
            if isinstance(value, dict):
                recursive_remove_save_location(value)
            elif isinstance(value, list):
                recursive_remove_save_location(value)


@pytest.mark.parametrize(
    [
        "verbosity",
        "basic_agent_definition",
        "environment",
        "number_of_timesteps",
        "number_of_episodes",
        "validation",
        "n_environments",
        "normalize_training_values",
        "multi_env_sequential",
    ],
    [
        (
            None,
            "PPO",
            None,
            10,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            None,
            "PPO",
            None,
            10,
            None,
            {
                "validation_values": [120],
                "validation_freq": 1,
                "save_best_agent": False,
                "validate_on_training_end": False,
            },
            None,
            None,
            None,
        ),
    ],
    indirect=["basic_agent_definition"],
)
def test_base_parser_validation(
    verbosity,
    basic_agent_definition,
    environment,
    number_of_timesteps,
    number_of_episodes,
    validation,
    n_environments,
    normalize_training_values,
    multi_env_sequential,
    dir_save_location,
    basic_environment_definition,
    basic_verbosity_definition,
    clean_up_provider,
):
    calling_dict = {
        "save_location": dir_save_location,
        "verbosity": basic_verbosity_definition,
        "agent": basic_agent_definition,
        "environment": basic_environment_definition,
        "number_of_timesteps": number_of_timesteps,
    }
    if verbosity is not None:
        calling_dict["verbosity"].update(verbosity)
    if environment is not None:
        calling_dict["environment"].update(environment)
    if number_of_episodes is not None:
        calling_dict["number_of_episodes"] = number_of_episodes
    if validation is not None:
        calling_dict["validation"] = validation
    if n_environments is not None:
        calling_dict["n_environments"] = n_environments
    if normalize_training_values is not None:
        calling_dict["normalize_training_values"] = normalize_training_values
    if multi_env_sequential is not None:
        calling_dict["multi_env_sequential"] = multi_env_sequential

    # recursive_remove_save_location(calling_dict)
    # assert "save_location" not in calling_dict["environment"]
    # if this does not fail, the save_location is recursively added as it
    # should be
    base_parser = BaseParser(**calling_dict)

    file_paths = []
    with pytest.raises(RuntimeError) as err:
        file_paths.append(base_parser.save_model())
    assert "Please train the agent first" in str(err.value)
    if validation is None:
        with pytest.raises(ValidationNotSet) as err:
            base_parser.evaluate_model(throw_error_if_none=True)

        with pytest.raises(ValidationNotSet) as err:
            base_parser._create_validation_environment(
                throw_error_if_none=True
            )

        clean_up_provider(dir_save_location)
        return
    base_parser.evaluate_model()

    file_paths.extend((
        base_parser.save_model(),
        base_parser.save_model("testing"),
    ))

    for file_path in file_paths:
        print(file_path)
        file_path = pathlib.Path(file_path)
        assert file_path.exists()
        clean_up_provider(
            file_path.parent
        )  # be very careful here that it does not delete more than you want

    clean_up_provider(dir_save_location)


@pytest.mark.parametrize(
    [
        "verbosity",
        "basic_agent_definition",
        "agent_additions",
        "environment",
        "number_of_timesteps",
        "number_of_episodes",
        "validation",
        "n_environments",
        "normalize_training_values",
        "multi_env_sequential",
        "export_step_log",
    ],
    [
        (
            None,
            "PPO",
            {
                "n_steps": 64,
            },
            None,
            10,
            None,
            None,
            5,
            None,
            False,
            False,
        ),
        (
            None,
            "PPO",
            {
                "n_steps": 64,
            },
            None,
            10,
            5,
            None,
            5,
            True,
            None,
            False,
        ),
        (
            None,
            "PPO",
            {
                "n_steps": 64,
            },
            None,
            10,
            None,
            None,
            5,
            None,
            True,
            True,
        ),
        (
            None,
            "PPO",
            {
                "n_steps": 64,
            },
            None,
            10,
            None,
            None,
            5,
            True,
            False,
            False,
        ),
        (
            None,
            "PPO",
            {
                "n_steps": 64,
            },
            None,
            10,
            None,
            {
                "validation_values": [120],
                "validation_freq": 1,
                "save_best_agent": False,
                "validate_on_training_end": False,
            },
            None,
            None,
            None,
            False,
        ),
    ],
    indirect=["basic_agent_definition"],
)
def test_base_parser_learn(
    verbosity,
    basic_agent_definition,
    agent_additions,
    environment,
    number_of_timesteps,
    number_of_episodes,
    validation,
    n_environments,
    normalize_training_values,
    multi_env_sequential,
    export_step_log,
    dir_save_location,
    basic_environment_definition,
    basic_verbosity_definition,
    clean_up_provider,
):
    calling_dict = {
        "save_location": dir_save_location,
        "verbosity": basic_verbosity_definition,
        "agent": basic_agent_definition,
        "environment": basic_environment_definition,
        "number_of_timesteps": number_of_timesteps,
        "export_step_log": export_step_log,
    }
    if agent_additions is not None:
        calling_dict["agent"].update(agent_additions)
    if verbosity is not None:
        calling_dict["verbosity"].update(verbosity)
    if environment is not None:
        calling_dict["environment"].update(environment)
    if number_of_episodes is not None:
        calling_dict["number_of_episodes"] = number_of_episodes
    if validation is not None:
        calling_dict["validation"] = validation
    if n_environments is not None:
        calling_dict["n_environments"] = n_environments
    if normalize_training_values is not None:
        calling_dict["normalize_training_values"] = normalize_training_values
    if multi_env_sequential is not None:
        calling_dict["multi_env_sequential"] = multi_env_sequential

    # recursive_remove_save_location(calling_dict)
    # assert "save_location" not in calling_dict["environment"]
    # if this does not fail, the save_location is recursively added as it
    # should be
    base_parser = BaseParser(**calling_dict)
    base_parser.learn()
    clean_up_provider(dir_save_location)
