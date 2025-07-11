import logging

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from releso.feature_extractor import CombinedExtractor, FeatureExtractor


def observation_space():
    return spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.uint8)


@pytest.mark.torch_test
@pytest.mark.parametrize(
    [
        "observation_space",
        "feature_dim",
        "without_linear",
        "network_type",
        "logger",
        "error",
    ],
    [
        (
            observation_space(),
            None,
            None,
            None,
            logging.getLogger("test"),
            False,
        ),
        (observation_space(), None, None, "mobilenet_v2", None, False),
        (
            observation_space(),
            None,
            None,
            "inceptionv3",
            logging.getLogger("test"),
            "Given network type -inceptionv3- unknown.",
        ),
        (observation_space(), None, None, "resnet18", None, False),
        (observation_space(), None, True, "resnet18", None, False),
    ],
)
def test_feature_extractor(
    observation_space,
    feature_dim,
    without_linear,
    network_type,
    logger,
    error,
):
    calling_dict = {
        "observation_space": observation_space,
    }
    if feature_dim is not None:
        calling_dict["feature_dim"] = feature_dim
    else:
        feature_dim = 128
    if without_linear is not None:
        calling_dict["without_linear"] = without_linear
    if network_type is not None:
        calling_dict["network_type"] = network_type
    if logger is not None:
        calling_dict["logger"] = logger
    if error:
        with pytest.raises(RuntimeError) as error_info:
            model = FeatureExtractor(**calling_dict)
        assert error in str(error_info.value)
        return
    model = FeatureExtractor(**calling_dict)
    result = model.forward(
        th.as_tensor(observation_space.sample()[None]).float()
    )
    if without_linear:
        assert model.features_dim == result.shape[1]
    else:
        assert model.features_dim == feature_dim == result.shape[1]


@pytest.mark.torch_test
@pytest.mark.parametrize(
    [
        "observation_space",
        "feature_dim",
        "without_linear",
        "network_type",
        "logger",
        "error",
    ],
    [
        (
            observation_space(),
            None,
            None,
            None,
            logging.getLogger("test"),
            False,
        ),
    ],
)
def test_combined_feature_extractor(
    observation_space,
    feature_dim,
    without_linear,
    network_type,
    logger,
    error,
):
    observation_space = spaces.Dict({
        "image": observation_space,
        "vector": spaces.Box(low=-1, high=1, shape=(10,), dtype=float),
    })

    calling_dict = {
        "observation_space": observation_space,
    }
    if feature_dim is not None:
        calling_dict["cnn_output_dim"] = feature_dim
    else:
        feature_dim = 128
    if without_linear is not None:
        calling_dict["without_linear"] = without_linear
    if network_type is not None:
        calling_dict["network_type"] = network_type
    if logger is not None:
        calling_dict["logger"] = logger
    if error:
        with pytest.raises(RuntimeError) as error_info:
            model = FeatureExtractor(**calling_dict)
        assert error in str(error_info.value)
        return
    model = CombinedExtractor(**calling_dict)
    _ = model.forward({
        key: th.as_tensor(value[None]).float()
        for key, value in observation_space.sample().items()
    })
