"""Non SB3 Callbacks used in ReLeSO.

Defines the custom call backs the ReLeSO library uses. Works now with
Multi-Environment training and should work with all agents.
"""
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeLogCallback(BaseCallback):
    """Episode Callback class, even for vectorized environments."""

    def __init__(
        self,
        episode_log_location: Path,
        verbose: int = 0,
        update_n_episodes: int = 1,
    ):
        """Constructor for the Callback using SB3 interface.

        Args:
            episode_log_location (Path): Path to the episode log.
            verbose (int, optional): Verbosity of the callback. Defaults to 0.
            update_n_episodes (int, optional): Update the episode every n
            episodes. Defaults to 1.
        """
        super().__init__(verbose=verbose)
        self.episode_log_location: Path = episode_log_location
        self.episode_log_location.parent.mkdir(parents=True, exist_ok=True)
        self.update_n_episodes: int = update_n_episodes

        self.episodes: int = -1
        self.last_checked_episode: int = 0

        self.inter_episode_dicts: Optional[
            List[Dict[int, Union[List[float], int]]]
        ] = None

        # concatenating the dataframe should be faster than appending to it
        self.last_exported_episode: int = -1

        # pandas dataframe do not seem to actually have a method to append new
        #  data to the dataframe without knowing the index. (append exists but
        #  is deprecated) Has probably something to do with allocating new
        #  memory and pandas not wanting to actually be used for this use case.
        self.episode_rewards: List[List[float]] = []
        self.episode_n_steps: List[int] = []
        self.episode_steps_total: List[int] = []
        self.episode_wall_time: List[str] = []
        self.episode_end: List[Optional[str]] = []
        self.environment_id: List[int] = []

    def _export(self):
        """Function exports the current information into a csv file."""
        export_data_frame = pd.DataFrame(
            {
                "steps_in_episode": self.episode_n_steps,
                "episode_reward": self.episode_rewards,
                "episode_end_reason": self.episode_end,
                "total_timesteps": self.episode_steps_total,
                "environment_id": self.environment_id,
                "wall_time": self.episode_wall_time,
            }
        )
        # export_data_frame.to_csv(self.episode_log_location)
        # export_data_frame.reset_index(drop=True, inplace=True)
        export_data_frame.index = np.arange(
            self.last_exported_episode + 1,
            self.last_exported_episode + 1 + len(export_data_frame),
        )
        if self.last_exported_episode == -1:
            export_data_frame.to_csv(self.episode_log_location)
        else:
            export_data_frame.to_csv(
                self.episode_log_location, mode="a", header=False
            )
        self.last_exported_episode = self.episodes
        # empty lists
        self.episode_n_steps = []
        self.episode_rewards = []
        self.episode_end = []
        self.episode_steps_total = []
        self.environment_id = []
        self.episode_wall_time = []

    def _on_step(self) -> bool:
        """Function is called after a step was performed.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """
        # first time setup of local vars for each environment
        if self.inter_episode_dicts is None:
            self.inter_episode_dicts: List[
                Dict[str, Union[List[float], int]]
            ] = []
            for _ in range(len(self.locals["dones"])):
                self.inter_episode_dicts.append(
                    {"current_episode_rewards": [], "last_start_step": 0}
                )
        continue_training = True
        for idx, loc_vars, done, reward, info in zip(
            count(),
            self.inter_episode_dicts,
            self.locals["dones"],
            self.locals["rewards"],
            self.locals["infos"],
        ):
            loc_vars["current_episode_rewards"].append(reward)
            if done:
                self.episodes += 1
                self.episode_rewards.append(
                    sum(loc_vars["current_episode_rewards"])
                )
                loc_vars["current_episode_rewards"] = []
                self.episode_n_steps.append(
                    self.n_calls - loc_vars["last_start_step"]
                )
                loc_vars["last_start_step"] = self.n_calls + 1
                self.episode_steps_total.append(self.num_timesteps)
                self.episode_wall_time.append(str(datetime.now()))
                reset_reason = (
                    None
                    if "reset_reason" not in info
                    else info["reset_reason"]
                )
                self.episode_end.append(reset_reason)
                self.environment_id.append(idx)
                if reset_reason == "srunError-main_solver":  # pragma: no cover
                    continue_training = False
        if any(
            episode % self.update_n_episodes == 0
            for episode in range(self.last_checked_episode, self.episodes)
        ):
            self._export()
        self.last_checked_episode = self.episodes
        return continue_training

    def _on_training_end(self) -> None:
        """Function is called when training is terminated."""
        self._export()
