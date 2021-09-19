from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
from pathlib import Path



class EpisodeLogCallback(BaseCallback):
    """Will most likely break with vectorized environment where training is performed on multiple environments concurrently.
    """
    def __init__(self, episode_log_location: Path, verbose: int = 0, update_n_episodes: int = 1):
        super().__init__(verbose=verbose)
        self.episodes = 0
        self.current_episode_rewards = []
        self.episode_rewards = []
        self.episode_n_steps = []
        self.episode_steps_total = []
        self.episode_wall_time = []
        self.last_end_step = 0
        self.episode_log_location = episode_log_location
        self.episode_log_location.parent.mkdir(parents=True, exist_ok=True)
        self.update_n_episodes = update_n_episodes

    def _export(self):
        df = pd.DataFrame({
            "steps_in_episode": self.episode_n_steps,
            "episode_reward": self.episode_rewards,
            "total_timesteps": self.episode_steps_total,
            "wall_time": self.episode_wall_time
        })
        df.to_csv(self.episode_log_location)

    def _on_step(self) -> bool:
        self.current_episode_rewards.append(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.episodes += 1
            self.episode_rewards.append(sum(self.current_episode_rewards))
            self.current_episode_rewards = []
            self.episode_n_steps.append(self.num_timesteps-self.last_end_step)
            self.last_end_step = self.num_timesteps
            self.episode_steps_total.append(self.num_timesteps)
            self.episode_wall_time.append(str(datetime.now()))
        if self.episodes % self.update_n_episodes == 0:
            self._export()
        return True

    def _on_training_end(self) -> None:
        self._export()
        print(self.locals)