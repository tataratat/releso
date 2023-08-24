import builtins
import os
import pathlib

import gymnasium as gym
import pytest
import requests

from releso.shape_parameterization import VariableLocation


class Dummy_Environment(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.episode_length = 10
        self.episode_counter = 0

    def step(self, action):
        self.episode_counter += 1
        return (
            [sum(action)],
            self.episode_counter,
            self.episode_counter >= self.episode_length,
            False,
            {},
        )

    def reset(self, **kwargs):
        self.episode_counter = 0
        return [0], {}


@pytest.fixture
def default_shape(dir_save_location):
    return {
        "control_points": [
            [
                1.0,
                2.0,
                3.0,
                4.0,
                VariableLocation(
                    **{
                        "current_position": 5.0,
                        "min_value": 4.0,
                        "save_location": dir_save_location,
                    }
                ),
            ],
            [
                1.0,
                2.0,
                3.0,
                4.0,
                {
                    "current_position": 5.0,
                    "min_value": 4.0,
                    "save_location": dir_save_location,
                },
            ],
        ],
        "save_location": dir_save_location,
    }


@pytest.fixture
def bspline_shape(dir_save_location):
    ret_dict = {
        "save_location": dir_save_location,
    }
    ret_dict["space_dimensions"] = [
        {
            "name": "something",
            "save_location": dir_save_location,
            "number_of_points": 3,
            "degree": 1,
        },
        {
            "name": "something",
            "save_location": dir_save_location,
            "number_of_points": 3,
            "degree": 1,
        },
    ]
    return ret_dict


@pytest.fixture
def nurbs_shape(bspline_shape, dir_save_location):
    bspline_shape["weights"] = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        VariableLocation(
            **{
                "current_position": 0.7,
                "max_value": 0.75,
                "min_value": 0.65,
                "save_location": dir_save_location,
            }
        ),
        VariableLocation(
            **{
                "current_position": 0.8,
                "max_value": 0.85,
                "min_value": 0.75,
                "save_location": dir_save_location,
            }
        ),
        0.9,
    ]
    return bspline_shape


@pytest.fixture
def provide_dummy_environment():
    return Dummy_Environment()


@pytest.fixture
def clean_up_provider():
    def recursive_file_remove(path):
        """Remove a file or directory and its contents.

        Very dangerous function. Use with the greatest of care.

        Author: GitHubCopilot (14.08.2023)
        """
        if not path.exists():
            return
        if path.is_file():
            path.unlink()
            return
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                recursive_file_remove(child)
        path.rmdir()

    return recursive_file_remove


@pytest.fixture
def hide_available_import(monkeypatch):
    """Hide the available import from the user.

    This is used to test the import of the available module.

    Author:
        https://stackoverflow.com/a/60229056
    """
    import_orig = builtins.__import__

    def mock_import_available(name, *args, **kwargs):
        # raise RuntimeError(name)
        with open("import.txt", "a") as file:
            file.write(f"{name}\n")
        if name == "splinepy.helpme.ffd":
            raise ImportError()
        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import_available)


def dir_save_location_path(make_dir=False):
    dir_save_location = (
        pathlib.Path(__file__).parent / "test_save_location_please_delete"
    ).resolve()
    if make_dir:
        dir_save_location.mkdir(parents=True, exist_ok=True)
    return dir_save_location


@pytest.fixture
def dir_save_location(clean_up_provider):
    path = dir_save_location_path()
    yield path
    clean_up_provider(path)


@pytest.fixture
def load_sample_file(request):
    file_name = request.param
    base_url = "https://raw.githubusercontent.com/tataratat/samples/main/"
    local_path = pathlib.Path(__file__).parent / "samples/"
    local_file = local_path / file_name
    local_file.parent.mkdir(parents=True, exist_ok=True)
    if not local_file.is_file():
        url = base_url + file_name
        # print(f"Downloading {url} to {local_file}")
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Could not download {url}")
        with open(local_file, "wb") as file:
            file.write(response.content)
    return local_file


@pytest.fixture
def dummy_file():
    file_name = pathlib.Path("dummy_file_please_delete.txt")
    file_name.touch()
    yield file_name
    if file_name.is_file():
        file_name.unlink()


@pytest.fixture
def basic_agent_definition(request, dir_save_location):
    return {
        "save_location": dir_save_location,
        "type": request.param,
        "policy": "MlpPolicy",
    }


@pytest.fixture
def basic_geometry_definition(dir_save_location, default_shape):
    return {
        "save_location": dir_save_location,
        "shape_definition": default_shape,
    }


@pytest.fixture
def basic_spor_definition(dir_save_location):
    dir_save_location_path(make_dir=True)
    return {
        "save_location": dir_save_location,
        "working_directory": str(dir_save_location),
        "name": "test",
        "reward_on_error": -1,
    }


@pytest.fixture
def spor_python_external(basic_spor_definition):
    file_path_prefix = str(pathlib.Path(__file__).parent)
    basic_spor_definition.update(
        {
            "python_file_path": (
                f"{file_path_prefix}/samples/spor_python_scripts_tests/"
                "file_exists_has_main.py"
            ),
            "use_communication_interface": True,
            "additional_observations": 3,
        }
    )
    return basic_spor_definition


@pytest.fixture
def basic_spor_list_definition(dir_save_location, spor_python_external):
    return {
        "save_location": dir_save_location,
        "steps": [spor_python_external],
        "reward_aggregation": "sum",
    }


@pytest.fixture
def basic_environment_definition(
    basic_geometry_definition, basic_spor_list_definition, dir_save_location
):
    return {
        "save_location": dir_save_location,
        "multi_processing": {
            "save_location": dir_save_location,
        },
        "geometry": basic_geometry_definition,
        "spor": basic_spor_list_definition,
    }


@pytest.fixture
def basic_verbosity_definition(dir_save_location):
    return {
        "save_location": dir_save_location,
    }
