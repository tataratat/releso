import numpy as np
import pytest
from gymnasium import spaces

from releso.observation import (
    ObservationDefinition,
    ObservationDefinitionMulti,
)


@pytest.mark.parametrize(
    "name, value_min, value_max, expected",
    [
        ("test", -1, 1, spaces.Box(-1, 1, shape=(1,), dtype=np.float32)),
        ("test", 0, 12, spaces.Box(0, 12, shape=(1,), dtype=np.float32)),
    ],
)
def test_observation_definition(
    name, value_min, value_max, expected, dir_save_location
):
    observation = ObservationDefinition(
        name=name,
        value_min=value_min,
        value_max=value_max,
        save_location=dir_save_location,
    )
    assert observation.get_observation_definition()[1] == expected
    assert np.allclose(
        observation.get_default_observation(), np.array([value_min])
    )


@pytest.mark.parametrize(
    "name, value_min, value_max, observation_shape, value_type, expected",
    [
        (
            "test",
            -1,
            1,
            [1],
            "float",
            spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        ),
        (
            "test",
            0,
            12,
            [1],
            "float",
            spaces.Box(0, 12, shape=(1,), dtype=np.float32),
        ),
        (
            "test",
            0,
            12,
            [12, 12],
            "float",
            spaces.Box(0, 12, shape=(12, 12), dtype=np.float32),
        ),
        (
            "test",
            0,
            255,
            [224, 224],
            "CNN",
            spaces.Box(0, 255, shape=(224, 224), dtype=np.uint8),
        ),
    ],
)
def test_multi_observation_definition(
    name,
    value_min,
    value_max,
    observation_shape,
    value_type,
    expected,
    dir_save_location,
):
    observation = ObservationDefinitionMulti(
        name=name,
        value_min=value_min,
        value_max=value_max,
        observation_shape=observation_shape,
        value_type=value_type,
        save_location=dir_save_location,
    )
    assert observation.get_observation_definition()[1] == expected

    if value_type == "float":
        assert np.allclose(
            np.ones(observation_shape) * value_min,
            observation.get_default_observation(),
        )
    elif value_type == "CNN":
        assert np.allclose(
            np.zeros(observation_shape, np.uint8),
            observation.get_default_observation(),
        )
    else:
        RuntimeError("Unknown value type")
