"""Classes used to parse the environment definition.

Files defines the environment used for parsing and also most function which
hold the functionality for the Reinforcement Learning environment are
defined here.
"""

import multiprocessing
import pathlib
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import gymnasium
import numpy as np
from gymnasium import spaces
from pydantic import UUID4, conint
from pydantic.class_validators import validator
from pydantic.fields import PrivateAttr
from stable_baselines3.common.monitor import Monitor

from releso.base_model import BaseModel
from releso.exceptions import ParserException
from releso.geometry import GeometryTypes
from releso.gym_environment import GymEnvironment
from releso.spor import SPORList
from releso.util.logger import VerbosityLevel, get_parser_logger, set_up_logger
from releso.util.types import ObservationType


class MultiProcessing(BaseModel):
    """Define multiprocessing capability of environment.

    Defines if the Problem should use Multiprocessing and with how many
    cores the solver can work. Does not force Multiprocessing for example
    if the solver does not support it.
    """

    #: Maximal number of cores which can be used by the current environment.
    #: Multi-Environments will use multiple of these.
    number_of_cores: conint(ge=1) = 1


class Environment(BaseModel):
    """Parser class of which the environment is based.

    Parser Environment object is created by pydantic during the parsing of the
    json object defining the RL based Shape optimization. Each object can
    create a gym environment that represents the given problem.
    """

    #: defines if multi-processing can be used.
    multi_processing: Optional[MultiProcessing]
    #: definition of the Geometry
    geometry: GeometryTypes
    #: definition of the spor objects
    spor: SPORList
    #: maximal number of timesteps to run each episode for
    max_timesteps_in_episode: Optional[conint(ge=1)] = None
    #: whether or not to reset the environment if the geometry has not change
    #: after a step
    end_episode_on_geometry_not_changed: bool = False
    #: reward if episode is ended due to reaching max step in episode
    reward_on_geometry_not_changed: Optional[float] = None
    #: reward if episode is ended due to geometry not changed
    reward_on_episode_exceeds_max_timesteps: Optional[float] = None
    # #: periodically save the end result of the optimization T-junction use
    # #: case TODO the next few cases I see personally more in a separate SPOR
    # #: Step but I am not sure how this can be worked since it needs data from
    # #: other sources
    # save_good_episode_results: bool = False
    # #: periodically save the end result of the optimization converging
    # #: channel use case
    # save_random_good_episode_results: bool = False
    # #: also saves the mesh when an episode is saved with
    # #: save_random_good_episode_results, works only if named option is True
    # save_random_good_episode_mesh: bool = False

    # object variables
    #: id if the environment, important for multi-environment learning
    _id: Optional[UUID4] = PrivateAttr(default=None)
    #: if validation environment validation ids are stored here
    _validation_ids: Optional[List[float]] = PrivateAttr(default=None)
    #: id of the current validation id
    _current_validation_idx: Optional[int] = PrivateAttr(default=None)
    #: number of timesteps currently spend in the episode
    _timesteps_in_episode: Optional[int] = PrivateAttr(default=0)
    #: StepReturn values from last step
    _last_step_results: Dict[str, Any] = PrivateAttr(default={})
    #: Toggle to whether or not it is possible to flatten the observation
    #: space. If the observation space is flattened the agents feature
    #: extractor is more compact
    _flatten_observations: bool = PrivateAttr(default=False)

    # The following are all variables of validation steps that are currently
    #  not implemented. These commented steps are still present due to the
    #  possibility of them being added again in the future. The optimal
    #  solution would be to add them as optional SPOR steps. This would be
    #  difficult due to limited information in each spor step.
    ##
    # #: path where the mesh should be saved to for validation
    # _validation_base_mesh_path: Optional[str] = PrivateAttr(default=None)
    # #: How many validations were already evaluated
    # _validation_iteration: Optional[int] = PrivateAttr(default=0)
    # #: Please check validation definition to see what this does
    # _save_image_in_validation: bool = PrivateAttr(default=False)
    # #: Please check validation definition to see what this does
    # _validation_timestep: int = PrivateAttr(default=0)
    # #: At these values the validation results are saved
    # _result_values_to_save: List[float] = PrivateAttr(
    #     default=list([0.2, 0.6, 0.9, 1.4, 1.8])
    # )
    # #: Approximate validation episode can be slightly wrong
    # _approximate_episode: int = PrivateAttr(default=-1)
    # #: Number of validation results which are already exported please check
    # #: function body to see/set the limit
    # _number_exported: int = PrivateAttr(default=0)

    @validator("reward_on_geometry_not_changed", always=True)
    @classmethod
    def check_if_reward_given_if_geometry_not_change_episode_killer_activated(
        cls, value, values
    ) -> float:
        """Validator reward_on_geometry_not_changed.

        Checks that 1) if a reward is set, also the boolean value for the
        end_episode_on_geometry_not_changed is True. 2) If
        end_episode_on_geometry_not_changed is True a reward value is set.

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values
                    (here end_episode_on_geometry_not_changed is important)

        Raises:
            ParserException: Error is thrown if one of the conditions is
            not met.

        Returns:
            float: reward for the specified occurrence.
        """
        # due to pydantic this first statement should never be able to happen
        if (
            "end_episode_on_geometry_not_changed" not in values
        ):  # pragma: no cover
            raise ParserException(
                "Environment",
                "reward_on_geometry_not_changed",
                "Could not find definition of parameter "
                "end_episode_on_geometry_not_changed, please defines this "
                "variable since otherwise this variable would have no "
                "function.",
            )
        if value is not None and (
            values["end_episode_on_geometry_not_changed"] is None
            or not values["end_episode_on_geometry_not_changed"]
        ):
            raise ParserException(
                "Environment",
                "reward_on_geometry_not_changed",
                "Reward can only be set if end_episode_on_geometry_not_changed"
                " is true.",
            )
        if values["end_episode_on_geometry_not_changed"] and value is None:
            get_parser_logger().warning(
                "Please set a reward value for geometry not changed if episode"
                " should end on it. Will set 0 for you now, but this might "
                "not be you intention."
            )
            value = 0.0
        return value

    @validator("reward_on_episode_exceeds_max_timesteps", always=True)
    @classmethod
    def check_if_reward_given_if_max_steps_killer_activated(
        cls, value, values
    ) -> int:
        """Validator reward_on_episode_exceeds_max_timesteps.

        Checks that 1) if a reward is set, also the boolean value for the
        max_timesteps_in_episode is True. 2) If max_timesteps_in_episode
        is True a reward value is set.

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values (here
            max_timesteps_in_episode is important)

        Raises:
            ParserException: Error is thrown if one of the conditions is
            not met.

        Returns:
            float: reward for the specified occurrence.
        """
        # due to pydantic this first statement should never be able to happen
        if "max_timesteps_in_episode" not in values:  # pragma: no cover
            raise ParserException(
                "Environment",
                "reward_on_episode_exceeds_max_timesteps",
                "Could not find definition of parameter "
                "max_timesteps_in_episode, please defines this variable since "
                "otherwise this variable would have no function.",
            )
        if value is not None and (
            values["max_timesteps_in_episode"] is None
            or not values["max_timesteps_in_episode"]
            or values["max_timesteps_in_episode"] <= 0
        ):
            raise ParserException(
                "Environment",
                "reward_on_episode_exceeds_max_timesteps",
                "Reward can only be set if max_timesteps_in_episode a positive"
                " integer.",
            )
        if values["max_timesteps_in_episode"] and value is None:
            get_parser_logger().warning(
                "Please set a reward value for max time steps exceeded, if "
                "episode should end on it. Will set 0 for you now, but this "
                "might not be your intention."
            )
            value = 0.0
        return value

    # object functions

    def __init__(self, **data: Any) -> None:
        """Construct the object."""
        super().__init__(**data)

    def _compress_observation_space_definition(
        self,
        observation_spaces: List[Tuple[str, ObservationType]],
        has_cnn_observations: bool,
    ) -> List[Tuple[str, ObservationType]]:
        """If possible compress observation space.

        If possible will compress the observation space into a single
        :py:`gymnasium.spaces.Box` observation space. This is not possible if
        cnn observations are being used and als not if the shape of the
        observations change between observations.

        Args:
            observation_spaces (List[Tuple[str,ObservationType]]):
                Uncompressed observations spaces.
            has_cnn_observations (bool): Observations include cnn observations

        Returns:
            List[Tuple[str,ObservationType]]: If possible compressed
                observation space if not unchanged observations space tuple
                list.
        """
        # check if there is potential for compressing the observation space
        if len(observation_spaces) > 1 and not has_cnn_observations:
            self.get_logger().info(
                "Found potential for compressed observation space."
            )
            new_space_min = []
            new_space_max = []
            new_space_shape = []
            for _, subspace in observation_spaces:
                new_space_min.extend(subspace.low)
                new_space_max.extend(subspace.high)
                new_space_shape.append(subspace.shape)
                if len(new_space_shape) > 1:
                    # if the shape do not match compression is not possible
                    if len(new_space_shape[-1]) != len(new_space_shape[-2]):
                        self.get_logger().info(
                            "Could not flatten observation space dict "
                            "definition into a single observation "
                            "space. Keeping dict definition."
                        )
                        break
                    else:
                        # adding up shapes
                        new_space_shape[0] = [
                            x + y
                            for x, y in zip(
                                new_space_shape[-1], new_space_shape[-2]
                            )
                        ]
                        new_space_shape.pop()
            else:
                # for loop completed so compression is possible
                self._flatten_observations = True
                observation_spaces = [
                    (
                        "flattened_observation_space",
                        spaces.Box(
                            low=np.array(new_space_min),
                            high=np.array(new_space_max),
                            shape=new_space_shape[0],
                        ),
                    )
                ]
        else:
            self.get_logger().info(
                "Did not find any potential for a "
                "compressed observation space."
            )

        return observation_spaces

    def _define_observation_space(self) -> gymnasium.Space:
        """Define the observation space of the environment.

        Creates the observation space the gym environment uses to define
        its observations space.

        Returns:
            # gymnasium.Space: Observation space of the current problem.
        """
        observation_spaces: List[Tuple[str, gymnasium.Space]] = []
        # define base observation
        if self.geometry.action_based_observation:
            observation_spaces.append(
                self.geometry.get_observation_definition()
            )
        # define spor observations
        has_cnn_observations = False
        spor_obs = self.spor.get_observations()
        if spor_obs is not None:
            for item in spor_obs:
                observation_spaces.append(item.get_observation_definition())
                if (
                    "value_type" in item.__dict__.keys()
                    and item.value_type == "CNN"
                ):
                    has_cnn_observations = True

        # check if dict is actually necessary
        observation_spaces = self._compress_observation_space_definition(
            observation_spaces, has_cnn_observations
        )

        if len(observation_spaces) > 1:
            # Observation space dict necessary
            observation_dict = dict(observation_spaces)
            self.get_logger().info(
                "Observation space is of type Dict and"
                " has the following description:"
            )
            for name, subspace in observation_spaces:
                self.get_logger().info(f"{name} has shape {subspace.shape}")
            return spaces.Dict(observation_dict)
        elif len(observation_spaces) == 1:
            # single observation space necessary
            self.get_logger().info(
                f"Observation space is NOT of type Dict and has the following"
                f" description: {observation_spaces[0][1].shape}"
            )
            # no dict space is needed so only the base observation space is
            # returned without a name
            return observation_spaces[0][1]
        else:
            raise RuntimeError(
                "The observation space is empty. Please define "
                "observations for RL to work."
            )

    def is_multiprocessing(self) -> int:
        """Check if environment uses multiprocessing.

        Function checks if the environment is setup to be used with
        multiprocessing Solver. Returns the number of cores the solver should
        use. If no multiprocessing 1 core is returned.

        Returns:
            int: Number of cores used in multiprocessing. 1 If no
                 multiprocessing. (Single thread still ok.)
        """
        return self.multi_processing.number_of_cores

    def get_validation_id(self) -> Optional[int]:
        """Return the validation id of the current run, if applicable.

        Checks if current environment has validation values if return the
        correct one otherwise return None.

        Returns:
            Optional[int]: Check text above.
        """
        if self._validation_ids is not None:
            return self._validation_ids[
                self._current_validation_idx % len(self._validation_ids)
            ]
        return None

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Performs the step of the environment.

        Function that is called for each step. Contains all steps that are
        performed during each step inside the environment.

        There was a change of what the step function returns. It now returns
        the following values:

        1. observation
        2. reward
        3. terminated
        4. truncated
        5. info

        The change was that done was split into terminated and truncated.
        Terminated is now True if the episode is done. Truncated is True if
        the episode was ended due to the maximum number of steps, gone outside
        of physical bounds or action values.

        This change is really interesting for us as it means that we might be
        able to handle the termination of the episode other than goal_states
        better. Currently truncated is set to False to make it compatible.
        Needs work to be put in to allow this information to propagate
        correctly.

        Args:
            action (Any): Action value depends on if the ActionSpace is
            discrete (int - Signifier of the action) or
            Continuous (List[float] - Value for each continuous
            variable.)

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]: [description]
        """
        start = timer()

        observations = {}
        done = False
        reward = 0.0
        info = {}

        # apply new action
        self.get_logger().info(f"Action {action}")
        info["geometry_information"] = self.geometry.apply_action(action)

        # run SPOR
        observations, reward, done, info = self.spor.run(
            step_information=(observations, done, reward, info),
            validation_id=self.get_validation_id(),
            core_count=self.is_multiprocessing(),
            reset=False,
            environment_id=self._id,
        )

        # check if geometry has not changed. But only in validation phase to
        #  exit episodes that are always repeating the same action without
        #  breaking.
        if not done:
            self._timesteps_in_episode += 1
            if (
                self.max_timesteps_in_episode
                and self.max_timesteps_in_episode > 0
                and self._timesteps_in_episode >= self.max_timesteps_in_episode
            ):
                done = True
                reward += self.reward_on_episode_exceeds_max_timesteps
                info["reset_reason"] = "max_timesteps_exceeded"
            if self.end_episode_on_geometry_not_changed:
                if not self.geometry.is_geometry_changed():
                    self.get_logger().info(
                        "The geometry observation have"
                        " not changed will exit episode."
                    )
                    reward += self.reward_on_geometry_not_changed
                    done = True
                    info["reset_reason"] = "geometry_not_changed"
        # else:
        #     pass  # TODO convert to separate SPOR Steps
        #     if reward >= 5.:
        #         if self.save_good_episode_results:
        #             for elem in self._result_values_to_save:
        #                 if np.isclose(
        #                     list(observations.values())[0][1],
        #                     elem, atol=0.05
        #                 ):
        #                     self.save_current_solution_as_png(
        #                         self.save_location/"episode_end_results"
        #                         ""/str(elem)/f""
        #                         f"{list(observations.values())[0][1]}"
        #                         f"_{self._approximate_episode}.png",
        #                         height=2, width=2, dpi=400)
        #                     break
        #         if self.save_random_good_episode_results:
        #             if self._number_exported < 200:
        #                 if np.random.default_rng().random() < 5:
        #                     self.save_current_solution_as_png(
        #                         self.save_location/"episode_end_results" /
        #                         f"{self._approximate_episode}.png",
        #                         height=2, width=2, dpi=400)
        #                     if self.save_random_good_episode_mesh:
        #                         self.export_mesh(
        #                             self.save_location/"episode_end_results"/
        #                             "mesh"/
        #                             f"{self._approximate_episode}.xns")
        #                     self._number_exported += 1

        self.get_logger().info(
            f"Current reward {reward} and episode is done: {done}."
        )

        self._last_step_results = {
            "observations": observations,
            "reward": reward,
            "done": done,
            "info": info,
        }
        self.get_logger().debug(self._last_step_results)
        # TODO convert to a SPOR Step
        # if self._validation_ids and self._save_image_in_validation:
        #     self._validation_timestep += 1
        #     self.save_current_solution_as_png(
        #         self.save_location/"validation"
        #         / str(self._validation_iteration)
        #         / str(self._current_validation_idx)
        #         / f"{self._validation_timestep}.png")

        observations["geometry_observation"] = self.geometry.get_observation()
        if observations["geometry_observation"] is None:
            del observations["geometry_observation"]
        end = timer()
        self.get_logger().debug(f"Step took {end - start} seconds.")

        return self.check_observations(observations), reward, done, False, info

    def check_observations(
        self, observations: ObservationType
    ) -> ObservationType:
        """Processes the observations.

        Transforms the observations from a dict observation to a normal
        observation if applicable.

        Args:
            observations (ObservationType): Observations to be processed.

        Returns:
            ObservationType: Processed Observations.
        """
        self.get_logger().debug(
            f"The observations are as follows: {observations}"
        )
        new_observation = []
        if len(observations.keys()) == 1:
            new_observation = observations[next(iter(observations.keys()))]
        else:
            if self._flatten_observations:
                for key in observations.keys():
                    new_observation.extend(observations[key])
            else:
                new_observation = observations
        return new_observation

    def reset(self, seed: Optional[int] = None) -> ObservationType:
        """Resets the environment.

        This can either be the case if the episode is done due to #time_steps
        or the environment emits the done signal.

        Args:
            seed (Optional[int], optional): Seed for the environment. Defaults
            to None.

        Returns:
            Tuple[Any]: Observation of the newly reset environment.
        """
        self.get_logger().info("Resetting the Environment.")

        observations = {}
        done = False
        reward = 0.0
        info = {}

        # reset geometry
        info["geometry_information"] = self.geometry.reset(
            self.get_validation_id()
        )

        self._timesteps_in_episode = 0
        # self._approximate_episode += 1
        # run solver and reset reward and get new solver observations
        observations, reward, done, info = self.spor.run(
            step_information=(observations, done, reward, info),
            validation_id=self.get_validation_id(),
            core_count=self.is_multiprocessing(),
            reset=True,
            environment_id=self._id,
        )

        if self._validation_ids:
            # TODO move to SPOR step
            # export mesh at end of validation
            # if self._current_validation_idx > 0 and \
            #         self._validation_base_mesh_path:
            #     # validation is performed in single environment
            #     # with no multi threading so this is not necessary.
            #     base_path = pathlib.Path(
            #         self._validation_base_mesh_path).parents[0]/str(
            #         self._validation_iteration)/str(
            #         self._current_validation_idx)
            #    file_name = pathlib.Path(self._validation_base_mesh_path).name
            #    if "_." in self._validation_base_mesh_path:
            #        validation_mesh_path = base_path / str(file_name).replace(
            #            "_.", f"{self.get_reset_reason_string()}.")
            #    else:
            #        validation_mesh_path = self._validation_base_mesh_path
            #    self.export_mesh(validation_mesh_path)
            #    self.export_spline(validation_mesh_path.with_suffix(".xml"))
            if self._current_validation_idx >= len(self._validation_ids):
                self.get_logger().info(
                    "The validation callback resets the environment one time "
                    "to often. Next goal state will again be the correct one."
                )
            self._current_validation_idx += 1
            if self._current_validation_idx > len(self._validation_ids):
                self._current_validation_idx = 0
                # self._validation_iteration += 1
        self.get_logger().info("Resetting the Environment DONE.")
        # TODO move to separate SPOR Step
        # if self._validation_ids and self._save_image_in_validation:
        #     self._validation_timestep = 0
        #     self.save_current_solution_as_png(
        #         self.save_location/"validation" /
        #         str(self._validation_iteration) /
        #         str(self._current_validation_idx) /
        #         f"{self._validation_timestep}.png")

        observations["geometry_observation"] = self.geometry.get_observation()
        if observations["geometry_observation"] is None:
            del observations["geometry_observation"]
        obs = self.check_observations(observations)
        return obs, info

    def set_validation(
        self,
        validation_values: List[float],
        end_episode_on_geometry_not_changed: bool = False,
        max_timesteps_in_episode: int = 0,
        reward_on_geometry_not_changed: Optional[float] = None,
        reward_on_episode_exceeds_max_timesteps: Optional[float] = None,
    ):
        """Converts the environment to a validation environment.

        This environment now only sets the goal states to the predefined
        values.

        Args:
            validation_values (List[float]): List of predefined goal states.
                base_mesh_path (Optional[str], optional): Path to the initial
                mesh. Defaults to None.
            end_episode_on_geometry_not_changed (bool, optional): Should the
                episode end if the geometry has no changes from one episode to
                the next. Defaults to False.
            max_timesteps_in_episode (int, optional): Maximal timesteps per
                episode, if 0 no limit. Defaults to 0.
            reward_on_geometry_not_changed (float, optional): Reward to give if
                episode is terminated due to unchanged geometry.
                Defaults to None.
            reward_on_episode_exceeds_max_timesteps(float, optional): Reward
                given of the episode is terminated due to exceeding the
                max_timesteps_in_episode. Defaults to None.
            save_image_in_validation (bool, optional): Should the validation
                save periodically the geometry/visualization. Is broken.
                Defaults to False.
        """
        self._validation_ids = validation_values
        self._current_validation_idx = 0
        # self._validation_base_mesh_path = base_mesh_path
        self.max_timesteps_in_episode = max_timesteps_in_episode
        self.end_episode_on_geometry_not_changed = (
            end_episode_on_geometry_not_changed
        )
        self.reward_on_geometry_not_changed = reward_on_geometry_not_changed
        self.reward_on_episode_exceeds_max_timesteps = (
            reward_on_episode_exceeds_max_timesteps
        )
        # self._save_image_in_validation = save_image_in_validation
        self.get_logger().info(
            f"Setting environment to validation. "
            f"max_timesteps {self.max_timesteps_in_episode}, "
            f"spine_not_changed {self.end_episode_on_geometry_not_changed}"
        )

    def get_gym_environment(
        self,
        logging_information: Optional[
            Dict[str, Union[str, pathlib.Path, VerbosityLevel]]
        ] = None,
    ) -> gymnasium.Env:
        """Creates the parametrized gymnasium environment.

        Creates and configures the gymnasium environment so it can be used
        for training.

        Returns:
            gymnasium.Env: OpenAI gymnasium environment that can be used to
                train with stable_baselines[3] agents.
        """
        if logging_information:
            self._set_up_logger(**logging_information)
        self.get_logger().info("Setting up Gym environment.")
        if self._id is None:
            self._id = uuid4()
        else:
            self.get_logger().warning(
                f"During setup of gym environment: The id of the environment "
                f"is already set to {self._id}. Please note that when using "
                "multi environment training this will lead to errors. Please "
                "use this function only after multiplying the environments "
                "and not before."
            )

        self.geometry.setup(self._id)
        env = GymEnvironment(
            self.geometry.get_action_definition(),
            self._define_observation_space(),
        )
        env.step = self.step
        env.reset = self.reset
        env.close = self.close
        return Monitor(env)

    def _set_up_logger(
        self,
        logger_name: str,
        log_file_location: pathlib.Path,
        logging_level: VerbosityLevel,
    ):
        """Setup the logger to be multiprocessing aware.

        Additional set up logger function for the purpose of multiprocessing
        capable logger initialization.

        Args:
            logger_name (str): name of the logger to set up
            log_file_location (pathlib.Path): location of the file where the
                logger should write to
            logging_level (VerbosityLevel): logger level to write up to
        """
        logger = multiprocessing.get_logger()
        resulting_logger = set_up_logger(
            logger_name, log_file_location, logging_level, logger=logger
        )
        self.set_logger_name_recursively(resulting_logger.name)

    def close(self):
        """Function is called when training is stopped."""
