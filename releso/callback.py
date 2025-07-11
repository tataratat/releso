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
        logger=None,
    ):
        """Constructor for the Callback using SB3 interface.

        Args:
            episode_log_location (Path): Path to the episode log.
            verbose (int, optional): Verbosity of the callback. Defaults to 0.
            update_n_episodes (int, optional): Update the episode every n
            episodes. Defaults to 1.
            logger (optional): Logger to use for logging. If None, a default
            logger is created. Defaults to None.
        """
        super().__init__(verbose=verbose)
        self.episode_log_location: Path = episode_log_location
        self.episode_log_location.parent.mkdir(parents=True, exist_ok=True)
        self.update_n_episodes: int = update_n_episodes

        self.episodes: int = -1
        self.last_checked_episode: int = 0
        self._logger = logger
        if self._logger is None:
            from stable_baselines3.common.logger import configure

            self._logger = configure(
                str(self.episode_log_location.parent / "episode_log.log"),
                ["stdout", "csv"],
            )

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
        export_data_frame = pd.DataFrame({
            "steps_in_episode": self.episode_n_steps,
            "episode_reward": self.episode_rewards,
            "episode_end_reason": self.episode_end,
            "total_timesteps": self.episode_steps_total,
            "environment_id": self.environment_id,
            "wall_time": self.episode_wall_time,
        })
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
                self.inter_episode_dicts.append({
                    "current_episode_rewards": [],
                    "last_start_step": 0,
                })
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


class StepLogCallback(BaseCallback):
    """Step Callback class.

    This class tracks all step-wise information that might come in handy
    during evaluation. Originally created to easily track the actions
    undertaken in each episode for better evaluation of the learned policy.
    """

    def __init__(
        self,
        step_log_location: Path,
        verbose: int = 0,
        update_n_steps: int = 0,
        log_infos: bool = False,
        logger=None,
    ):
        """Constructor for the Callback using SB3 interface.

        Args:
            step_log_location (Path): Path to the step log file.
            verbose (int, optional): Verbosity of the callback. Defaults to 0.
            update_every (int, optional): Update the step log file every n
            steps. Defaults to 0 which triggers the update after every episode.
            log_infos (bool, optional): Whether to log the infos. Defaults to
            False.
            logger (optional): Logger to use for logging. If None, a default
            logger is created. Defaults to None.
        """
        super().__init__(verbose)
        self.step_log_location: Path = step_log_location
        self.step_log_location.parent.mkdir(parents=True, exist_ok=True)
        self.current_episodes: list[int] = []
        self.update_n_episodes: int = update_n_steps
        self.first_export: bool = True
        self.current_max_episodes: int = 0
        self.log_infos: bool = log_infos

        self._logger = logger
        if self._logger is None:
            from stable_baselines3.common.logger import configure

            self._logger = configure(
                str(self.step_log_location.parent / "episode_log.log"),
                ["stdout", "csv"],
            )
        self._logger.info(
            (
                "The StepLogCallback should only be used for debugging purposes"
                " and only for use-cases with a smallish observation space."
                " Otherwise it will create a lot of data and slow down the"
                " training process. Use with caution!"
            )
        )
        self._reset_internal_storage()

    def _reset_internal_storage(self) -> None:
        """Reset the internally used lists which store the step-wise
        information since last updating the logfile.
        """
        self.timesteps = []  # Store step numbers
        self.episodes = []  # Store episode numbers
        self.dones = []  # Store if the episode has ended
        self.prev_obs = []  # Store observations at the start of time step
        self.actions = []  # Store actions
        self.new_obs = []  # Store observations after completion of time step
        self.rewards = []  # Optionally store rewards
        self.infos = []  # Optionally store additional information

    def _export(self) -> None:
        """Convert the step-wise information to a dataframe and export to csv."""
        # Combine all relevant information into a pandas DataFrame
        export_data_frame = pd.DataFrame({
            "episodes": self.episodes,
            "dones": self.dones,
            "prev_obs": self.prev_obs,
            "actions": self.actions,
            "new_obs": self.new_obs,
            "rewards": self.rewards,
        })
        if self.log_infos:
            export_data_frame["infos"] = self.infos
        export_data_frame.index = self.timesteps
        export_data_frame.index.name = "timesteps"
        # Write the data to file
        export_data_frame.to_json(
            self.step_log_location,
            orient="records",
            lines=True,
            mode="a" if not self.first_export else "w",
        )
        # data frame has been exported already at least once, so reset the flag
        self.first_export = False
        # reset the internal storage
        self._reset_internal_storage()

    def _on_step(self) -> bool:
        """Function that is called after a step was performed.

        Returns:
            bool: If the callback returns False, training is aborted early.
        """
        # Retrieve the step-wise information that we want to keep track of
        dones = self.locals[
            "dones"
        ]  # Field indicating if the episode has ended
        if len(self.current_episodes) == 0:
            n_dones = len(dones)
            self.current_episodes = [idx - n_dones for idx in range(n_dones)]
            self.current_max_episodes = max(self.current_episodes)
        prev_obs = self.model._last_obs  # Old observations
        actions = self.locals["actions"]  # Agent's actions
        new_obs = self.locals["new_obs"]  # New observations
        rewards = self.locals["rewards"]  # Rewards (optional)
        infos = self.locals["infos"]  # Additional information

        # Store additional information
        self.timesteps.append(self.num_timesteps)
        self.episodes.append(self.current_episodes)
        self.dones.append(dones)
        self.prev_obs.append(prev_obs)
        self.actions.append(actions)
        self.new_obs.append(new_obs)
        self.rewards.append(rewards)

        if self.log_infos:
            infos = self.locals["infos"]  # Additional information
            self.infos.append(infos)

        done_export = False
        # Check if the environment has completed an episode
        for idx, done in enumerate(dones):
            if done:
                # If the environment has completed an episode, locals["new_obs"]
                # will already contain the observation AFTER a reset has been
                # performed and not the original terminal observation with which the
                # episode ended, so overwrite the entries with the true terminal
                # observation
                self.new_obs[-1][idx] = self.locals["infos"][idx][
                    "terminal_observation"
                ]
                # If the update is supposed to be performed after an episode has
                # been completed ...
                if self.update_n_episodes == 0:
                    # schedule the export of the information
                    done_export = True
                # Always increase the episode counter
                self.current_episodes[idx] = self.current_max_episodes + 1
                self.current_max_episodes = self.current_episodes[idx]
        # If export scheduled or export with the given
        # frequency
        if done_export or (
            self.update_n_episodes != 0
            and any(
                timestep % self.update_n_episodes == 0
                for timestep in self.timesteps
            )
        ):
            self._export()

        return True
