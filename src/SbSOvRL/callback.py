# TODO overhall for multi environment learning
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from pathlib import Path
from typing import List, Optional



class EpisodeLogCallback(BaseCallback):
    """Will most likely break with vectorized environment where training is performed on multiple environments concurrently. Good for vectorized with only a single environment.
    """
    def __init__(self, episode_log_location: Path, verbose: int = 0, update_n_episodes: int = 1):
        super().__init__(verbose=verbose)
        self.episodes: int = 0
        self.current_episode_rewards: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_n_steps: List[int] = []
        self.episode_steps_total: List[int] = []
        self.episode_wall_time: List[str] = []
        self.episode_end: List[Optional[str]] = []
        self.last_end_step: Optional[int] = None
        self.episode_log_location: Path = episode_log_location
        self.episode_log_location.parent.mkdir(parents=True, exist_ok=True)
        self.update_n_episodes: int = update_n_episodes

    def _export(self):
        df = pd.DataFrame({
            "steps_in_episode": self.episode_n_steps,
            "episode_reward": self.episode_rewards,
            "epsiode_end_reason": self.episode_end,
            "total_timesteps": self.episode_steps_total,
            "wall_time": self.episode_wall_time
        })
        df.to_csv(self.episode_log_location)

    def _on_step(self) -> bool:
        # need this try because dqn is stupid
        if self.last_end_step is None:
            if self.num_timesteps <= 1:
                self.last_end_step = 0
            else:
                self.last_end_step = self.num_timesteps-1
        try:
            self.current_episode_rewards.append(self.locals["rewards"][0])
        except KeyError:
            self.current_episode_rewards.append(self.locals["reward"][0])
        try:
            done = self.locals["dones"][0]
        except KeyError:
            done = self.locals["done"][0]
        try:
            info = self.locals["info"][0]
        except KeyError:
            info = self.locals["infos"][0]
        if done:
            self.episodes += 1
            self.episode_rewards.append(sum(self.current_episode_rewards))
            self.current_episode_rewards = []
            self.episode_n_steps.append(self.num_timesteps-self.last_end_step)
            self.last_end_step = self.num_timesteps
            self.episode_steps_total.append(self.num_timesteps)
            self.episode_wall_time.append(str(datetime.now()))
            self.episode_end.append(None if "reset_reason" not in info.keys() else info["reset_reason"])
        if self.episodes % self.update_n_episodes == 0:
            self._export()
        return True

    def _on_training_end(self) -> None:
        self._export()