import pandas as pd
import pytest
from stable_baselines3 import PPO

from releso.callback import EpisodeLogCallback, StepLogCallback


def test_callback_episode_log_callback(
    dir_save_location,
    clean_up_provider,
    get_null_logger,
    provide_dummy_environment,
):
    # this test is not very good, but it is a start
    # TODO: improve this test
    call_back = EpisodeLogCallback(
        episode_log_location=dir_save_location / "test.csv",
        logger=get_null_logger,
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


@pytest.mark.parametrize(
    [
        "update_n_steps",
    ],
    [
        (0,),
        (20,),
    ],
)
def test_callback_step_information_log_callback(
    dir_save_location,
    clean_up_provider,
    get_null_logger,
    provide_dummy_environment,
    update_n_steps,
):
    # this test is not very good, but it is a start
    # TODO: improve this test
    call_back = StepLogCallback(
        step_log_location=dir_save_location / "test.jsonl",
        update_n_steps=update_n_steps,
        log_infos=True,
        logger=get_null_logger,
    )
    assert call_back.step_log_location == dir_save_location / "test.jsonl"
    assert call_back.current_episodes == []
    assert call_back.update_n_episodes == update_n_steps
    assert call_back.first_export
    env = provide_dummy_environment
    agent = PPO("MlpPolicy", env, verbose=0, n_steps=100)
    agent.learn(100, callback=call_back)
    episode_log = pd.read_json(
        dir_save_location / "test.jsonl",
        lines=True,
    )
    assert episode_log.shape[0] == 100
    clean_up_provider(dir_save_location)
