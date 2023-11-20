import pandas as pd
from stable_baselines3 import PPO

from releso.callback import EpisodeLogCallback


def test_callback_episode_log_callback(
    dir_save_location, clean_up_provider, provide_dummy_environment
):
    # this test is not very good, but it is a start
    # TODO: improve this test
    call_back = EpisodeLogCallback(
        episode_log_location=dir_save_location / "test.csv"
    )
    assert call_back.episode_log_location == dir_save_location / "test.csv"
    assert call_back.episodes == -1
    assert call_back.update_n_episodes == 1
    env = provide_dummy_environment
    agent = PPO("MlpPolicy", env, verbose=0, n_steps=100)
    agent.learn(100, callback=call_back)
    episode_log = pd.read_csv(dir_save_location / "test.csv")
    assert episode_log.shape[0] == 10
    clean_up_provider(dir_save_location)
