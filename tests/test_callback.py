import gymnasium as gym
import pandas as pd
import pytest
from stable_baselines3 import PPO

from releso.callback import EpisodeLogCallback


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


def test_callback_episode_log_callback(
    dir_save_location, clean_up_provider, caplog
):
    # this test is not very good, but it is a start
    # TODO: improve this test
    call_back = EpisodeLogCallback(
        episode_log_location=dir_save_location / "test.csv"
    )
    assert call_back.episode_log_location == dir_save_location / "test.csv"
    assert call_back.episodes == 0
    assert call_back.update_n_episodes == 1
    env = Dummy_Environment()
    agent = PPO("MlpPolicy", env, verbose=0, n_steps=100)
    agent.learn(100, callback=call_back)
    episode_log = pd.read_csv(dir_save_location / "test.csv")
    assert episode_log.shape[0] == 10
    clean_up_provider(dir_save_location)
