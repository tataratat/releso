import datetime

import gymnasium as gym
import pytest
from pydantic import BaseModel
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC

from releso.agent import (
    A2CAgent,
    AgentTypeDefinition,
    BaseAgent,
    BaseTrainingAgent,
    DDPGAgent,
    DQNAgent,
    PPOAgent,
    PretrainedAgent,
    SACAgent,
)
from releso.feature_extractor import CombinedExtractor, FeatureExtractor
from releso.verbosity import VerbosityLevel


@pytest.mark.parametrize(
    "log_name",
    [
        (None),
        (False),
        ("test"),
    ],
)
def test_base_agent_tensorboard_log(log_name, dir_save_location):
    calling_dict = {
        "save_location": dir_save_location,
    }
    if log_name is not False:
        calling_dict["tensorboard_log"] = log_name

    b_agent = BaseAgent(**calling_dict)

    if log_name:
        assert b_agent.get_next_tensorboard_experiment_name() == (
            f"{log_name}_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
    else:
        assert b_agent.get_next_tensorboard_experiment_name() is None


@pytest.mark.parametrize(
    "policy, use_custom_feature_extractor, cfe_without_linear, policy_kwargs",
    [
        ("MlpPolicy", None, False, None),
        ("MlpPolicy", None, True, {}),
        ("CnnPolicy", None, False, None),
        ("CnnPolicy", None, True, {}),
        ("MultiInputPolicy", None, False, {}),
        ("MultiInputPolicy", None, True, {}),
        ("MlpPolicy", "resnet18", True, {}),
        ("MlpPolicy", "resnet18", False, {}),
        ("CnnPolicy", "resnet18", False, {}),
        ("CnnPolicy", "resnet18", True, {}),
        ("MultiInputPolicy", "resnet18", False, {}),
        ("MultiInputPolicy", "resnet18", True, {}),
        ("MlpPolicy", "mobilenet_v2", True, {}),
        ("MlpPolicy", "mobilenet_v2", False, {}),
        ("CnnPolicy", "mobilenet_v2", False, {}),
        ("CnnPolicy", "mobilenet_v2", True, {}),
        ("MultiInputPolicy", "mobilenet_v2", False, {}),
        ("MultiInputPolicy", "mobilenet_v2", True, {}),
    ],
)
def test_base_training_agent(
    policy,
    use_custom_feature_extractor,
    cfe_without_linear,
    policy_kwargs,
    dir_save_location,
    caplog,
):
    calling_dict = {
        "save_location": dir_save_location,
        "policy": policy,
        "cfe_without_linear": cfe_without_linear,
        "policy_kwargs": policy_kwargs,
    }
    if use_custom_feature_extractor is not None:
        calling_dict["use_custom_feature_extractor"] = (
            use_custom_feature_extractor
        )
    base_training_agent = BaseTrainingAgent(**calling_dict)

    with caplog.at_level(VerbosityLevel.WARNING):
        additional_kwargs = base_training_agent.get_additional_kwargs()
    assert isinstance(additional_kwargs, dict)
    assert additional_kwargs["policy"] == policy
    if policy == "MlpPolicy" and use_custom_feature_extractor is not None:
        assert (
            "Please use the CnnPolicy or the MultiInputPolicy with"
            in caplog.text
        )
        assert (
            additional_kwargs["policy_kwargs"]["features_extractor_class"]
            == FeatureExtractor
        )
    if policy == "CnnPolicy" and use_custom_feature_extractor is not None:
        assert (
            additional_kwargs["policy_kwargs"]["features_extractor_class"]
            == FeatureExtractor
        )
    if (
        policy == "MultiInputPolicy"
        and use_custom_feature_extractor is not None
    ):
        assert (
            additional_kwargs["policy_kwargs"]["features_extractor_class"]
            == CombinedExtractor
        )
    if use_custom_feature_extractor is not None:
        if cfe_without_linear:
            assert (
                additional_kwargs["policy_kwargs"][
                    "features_extractor_kwargs"
                ]["without_linear"]
                is True
            )
        assert (
            additional_kwargs["policy_kwargs"]["features_extractor_kwargs"][
                "network_type"
            ]
            == use_custom_feature_extractor
        )
        assert (
            "logger"
            in additional_kwargs["policy_kwargs"][
                "features_extractor_kwargs"
            ].keys()
        )


class Dummy(BaseModel):
    agent: AgentTypeDefinition


@pytest.mark.parametrize("normalizer_divisor", [(1), (100), (0)])
@pytest.mark.parametrize(
    "agent, wanted_agent, resulting_agent",
    [
        ("PPO", PPOAgent, PPO),
        ("A2C", A2CAgent, A2C),
        ("DDPG", DDPGAgent, DDPG),
        ("DQN", DQNAgent, DQN),
        ("SAC", SACAgent, SAC),
    ],
)
def test_agents(
    agent,
    normalizer_divisor,
    wanted_agent,
    resulting_agent,
    dir_save_location,
    caplog,
):
    calling_dict = {
        "save_location": dir_save_location,
        "type": agent,
        "policy": "MlpPolicy",
    }
    ag = Dummy(agent=calling_dict)
    assert isinstance(ag.agent, wanted_agent)

    with caplog.at_level(VerbosityLevel.WARNING):
        if agent == "DQN":  # must use discrete actions
            agent = ag.agent.get_agent(
                gym.make("MountainCar-v0"), normalizer_divisor
            )
        else:
            agent = ag.agent.get_agent(
                gym.make("Pendulum-v1"), normalizer_divisor
            )
        if agent in ["PPO", "A2C"] and normalizer_divisor == 0:
            assert "Normalizer divisor is 0, will use 1." in caplog.text
            normalizer_divisor = 1

    assert isinstance(agent, resulting_agent)


@pytest.mark.parametrize(
    "agent, tensorboard_run_directory, tensorboard_log",
    [
        ("PPO", None, None),
        ("PPO", "test", None),
        ("PPO", None, "test"),
    ],
)
def test_pretrained_agent(
    agent,
    tensorboard_run_directory,
    tensorboard_log,
    dir_save_location,
    dummy_file,
):
    calling_dict = {
        "save_location": dir_save_location,
        "type": agent,
        "tensorboard_log": tensorboard_log,
        "path": dummy_file,
    }
    if tensorboard_run_directory:
        calling_dict["tesorboard_run_directory"] = tensorboard_run_directory
    ag = PretrainedAgent(**calling_dict)
    if tensorboard_run_directory:
        assert (
            ag.get_next_tensorboard_experiment_name()
            == tensorboard_run_directory
        )
    else:
        if tensorboard_log is not None:
            assert (
                ag.get_next_tensorboard_experiment_name()
                == f"{tensorboard_log}_"
                f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        else:
            assert ag.get_next_tensorboard_experiment_name() is None


@pytest.mark.skip(reason="no way of easily testing this")
def test_pretrained_load_correct_agent():
    pass
