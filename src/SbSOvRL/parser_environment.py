"""
    Files defines the environment used for parsing and also most function which
     hold the functionality for the Reinforcement Learning environment are
     defined here.
"""
import multiprocessing
import pathlib
import datetime
from typing import Optional, Any, List, Dict, Tuple, Union
import numpy as np
from copy import copy
from uuid import uuid4
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from pydantic.class_validators import validator
from pydantic.fields import Field, PrivateAttr
from pydantic import conint, UUID4

import gym
from gym import spaces
from stable_baselines3.common.monitor import Monitor

from gustav import FreeFormDeformation

from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.spline import Spline, VariableLocation
from SbSOvRL.util.logger import get_parser_logger
from SbSOvRL.mesh import Mesh
from SbSOvRL.spor import MultiProcessor, SPORList
from SbSOvRL.gym_environment import GymEnvironment
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.util.sbsovrl_types import ObservationType
from SbSOvRL.util.logger import VerbosityLevel, set_up_logger
from SbSOvRL.util.load_binary import read_mixd_double
from SbSOvRL.util.plotting import get_tricontour_solution


class MultiProcessing(SbSOvRL_BaseModel):
    """
        Defines if the Problem should use Multiprocessing and with how many
        cores the solver can work. Does not force Multiprocessing for example
        if the solver does not support it.
    """
    #: Maximal number of cores which can be used by the current environment.
    #: Multi-Environments will use multiple of these.
    number_of_cores: conint(ge=1) = 1


class Environment(SbSOvRL_BaseModel):
    """
    Parser Environment object is created by pydantic during the parsing of the
    json object defining the Spline base Shape optimization. Each object can
    create a gym environment that represents the given problem.
    """
    #: defines if multi-processing can be used.
    multi_processing: Optional[MultiProcessing] = Field(
        default_factory=MultiProcessor)
    spline: Spline  #: definition of the spline
    mesh: Mesh  #: definition of the mesh
    spor: SPORList  #: definition of the spor objects
    #: Whether or not to use discrete actions if False continuous actions
    #: will be used.
    discrete_actions: bool = True
    #: Whether or not to reset the controllable control point variables to a
    #: random state (True) or the default original state (False).
    #: Default False.
    reset_with_random_control_points: bool = False
    #: maximal number of timesteps to run each episode for
    max_timesteps_in_episode: Optional[conint(ge=1)] = None
    #: whether or not to reset the environment if the spline has not change
    #: after a step
    end_episode_on_spline_not_changed: bool = False
    #: reward if episode is ended due to reaching max step in episode
    reward_on_spline_not_changed: Optional[float] = None
    #: reward if episode is ended due to spline not changed
    reward_on_episode_exceeds_max_timesteps: Optional[float] = None
    #: use a cnn based feature extractor, if false the base observations will
    #: be supplied by the movable spline coordinates current values.
    use_cnn_observations: bool = False
    #: periodically save the end result of the optimization T-junction use case
    save_good_episode_results: bool = False
    #: periodically save the end result of the optimization converging channel
    #: use case
    save_random_good_episode_results: bool = False
    #: also saves the mesh when an episode is saved with
    #: save_random_good_episode_results, works only if the named option is True
    save_random_good_episode_mesh: bool = False

    # object variables
    #: id if the environment, important for multi-environment learning
    _id: UUID4 = PrivateAttr(default=None)
    _actions: List[VariableLocation] = PrivateAttr()    #: list of actions
    #: if validation environment validation ids are stored here
    _validation_ids: Optional[List[float]] = PrivateAttr(default=None)
    _current_validation_idx: Optional[int] = PrivateAttr(
        default=None)  #: id of the current validation id
    #: number of timesteps currently spend in the episode
    _timesteps_in_episode: Optional[int] = PrivateAttr(default=0)
    #: last observation from the last step used to determine whether or not the
    #: spline has changed between episodes
    _last_observation: Optional[ObservationType] = PrivateAttr(default=None)
    #: FreeFormDeformation used for the spline based shape optimization
    _FFD: FreeFormDeformation = PrivateAttr(
        default_factory=FreeFormDeformation)
    _last_step_results: Dict[str, Any] = PrivateAttr(
        default={})    #: StepReturn values from last step
    #: path where the mesh should be saved to for validation
    _validation_base_mesh_path: Optional[str] = PrivateAttr(default=None)
    _validation_iteration: Optional[int] = PrivateAttr(
        default=0)   #: How many validations were already evaluated
    _connectivity: Optional[np.ndarray] = PrivateAttr(
        default=None)  #: The triangular connectivity of the
    #: Please check validation definition to see what this does
    _save_image_in_validation: bool = PrivateAttr(default=False)
    #: Please check validation definition to see what this does
    _validation_timestep: int = PrivateAttr(default=0)
    _observation_is_dict: bool = PrivateAttr(default=False)
    #: At these values the validation results are saved
    _result_values_to_save: List[float] = PrivateAttr(default=list(
        [0.2, 0.6, 0.9, 1.4, 1.8]))
    #: Approximate validation episode can be slightly wrong
    _approximate_episode: int = PrivateAttr(default=-1)
    #: Number of validation results which are already exported please check
    #: function body to see/set the limit
    _number_exported: int = PrivateAttr(default=0)
    #: Toggle to whether or not it is possible to flatten the observation
    #: space. If the observation space is flattened the agents feature
    #: extractor is more compact
    _flatten_observations: bool = PrivateAttr(default=False)

    @validator("reward_on_spline_not_changed", always=True)
    @classmethod
    def check_if_reward_given_if_spline_not_change_episode_killer_activated(
            cls, value, values) -> float:
        """Checks that 1) if a reward is set, also the boolean value for the
        end_episode_on_spline_not_changed is True. 2) If
        end_episode_on_spline_not_changed is True a reward value is set.

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values
                    (here end_episode_on_spline_not_changed is important)

        Raises:
            SbSOvRLParserException: Error is thrown if one of the conditions is
             not met.

        Returns:
            float: reward for the specified occurrence.
        """
        if "end_episode_on_spline_not_changed" not in values:
            raise SbSOvRLParserException(
                "Environment", "reward_on_spline_not_changed",
                "Could not find definition of parameter "
                "end_episode_on_spline_not_changed, please defines this "
                "variable since otherwise this variable would have no "
                "function.")
        if value is not None and (values["end_episode_on_spline_not_changed"]
                                  is None or not
                                  values["end_episode_on_spline_not_changed"]):
            raise SbSOvRLParserException(
                "Environment", "reward_on_spline_not_changed",
                "Reward can only be set if end_episode_on_spline_not_changed "
                "is true.")
        if values["end_episode_on_spline_not_changed"] and value is None:
            get_parser_logger().warning(
                "Please set a reward value for spline not changed if episode "
                "should end on it. Will set 0 for you now, but this might "
                "not be you intention.")
            value = 0.
        return value

    @validator("reward_on_episode_exceeds_max_timesteps", always=True)
    @classmethod
    def check_if_reward_given_if_max_steps_killer_activated(cls, value,
                                                            values) -> int:
        """
            Checks that 1) if a reward is set, also the boolean value for the
            max_timesteps_in_episode is True. 2) If max_timesteps_in_episode
            is True a reward value is set.

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values (here
            max_timesteps_in_episode is important)

        Raises:
            SbSOvRLParserException: Error is thrown if one of the conditions is
            not met.

        Returns:
            float: reward for the specified occurrence.
        """
        if "max_timesteps_in_episode" not in values:
            raise SbSOvRLParserException(
                "Environment", "reward_on_episode_exceeds_max_timesteps",
                "Could not find definition of parameter "
                "max_timesteps_in_episode, please defines this variable since "
                "otherwise this variable would have no function.")
        if value is not None and (values["max_timesteps_in_episode"] is None
                                  or not values["max_timesteps_in_episode"]):
            raise SbSOvRLParserException(
                "Environment", "reward_on_episode_exceeds_max_timesteps",
                "Reward can only be set if max_timesteps_in_episode a positive"
                " integer.")
        if values["max_timesteps_in_episode"] and value is None:
            get_parser_logger().warning(
                "Please set a reward value for max time steps exceeded, if "
                "episode should end on it. Will set 0 for you now, but this "
                "might not be your intention.")
            value = 0.
        return value
    # object functions

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

    def _set_up_actions(self) -> gym.Space:
        """Creates the action space the gym environment uses to define its
            action space.

        Returns:
            gym.Space: action space of the current problem.
        """
        if self.discrete_actions:
            return spaces.Discrete(len(self._actions) * 2)
        else:
            return spaces.Box(low=-1, high=1, shape=(len(self._actions),))

    def _compress_observation_space_definition(
        self,
        observation_spaces: List[Tuple[str, ObservationType]],
        has_cnn_observations: bool
    ) -> List[Tuple[str, ObservationType]]:
        """
        If possible will compress the observation space into a single
        :py:`gym.spaces.Box` observation space. This is not possible if cnn
        observations are being used and als not if the shape of the
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
        if len(observation_spaces) > 1 and \
                not (self.use_cnn_observations or has_cnn_observations):
            self.get_logger().info("Found potential for compressed "
                                   "observation space.")
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
                            "definition into a single observation observation "
                            "space. Keeping dict definition.")
                        break
                    else:
                        # adding up shapes
                        new_space_shape[0] = (
                            x+y for x, y in zip(
                                new_space_shape[-1],
                                new_space_shape[-2]))
                        new_space_shape.pop()
            else:
                # for loop completed so compression is possible
                self._flatten_observations = True
                observation_spaces = [
                    ("flattened_observation_space",
                     spaces.Box(
                         low=np.array(new_space_min),
                         high=np.array(new_space_max),
                         shape=new_space_shape[0]
                     ))]
        else:
            self.get_logger().info("Did not find any potential for a "
                                   "compressed observation space.")

        return observation_spaces

    def _define_observation_space(self) -> gym.Space:
        """
            Creates the observation space the gym environment uses to define
            its observations space.

        Returns:
            # gym.Space: Observation space of the current problem.
        """
        # TODO This needs to be changed
        observation_spaces: List[Tuple[str, gym.Space]] = []
        # define base observation
        if self.use_cnn_observations:
            observation_spaces.append(("base_observation", spaces.Box(
                low=0, high=255, shape=(3, 200, 200), dtype=np.uint8)))
        else:
            observation_spaces.append(
                ("base_observation", spaces.Box(low=0, high=1,
                                                shape=(len(self._actions), ),
                                                dtype=np.float32)))

        # define spor observations
        has_cnn_observations = False
        spor_obs = self.spor.get_number_of_observations()
        if spor_obs is not None:
            for item in spor_obs:
                observation_spaces.append(item.get_observation_definition())
                if item.value_type == "CNN":
                    has_cnn_observations = True

        # check if dict is actually necessary
        observation_spaces = self._compress_observation_space_definition(
            observation_spaces, has_cnn_observations
        )

        if len(observation_spaces) > 1:
            # Observation space dict necessary
            self._observation_is_dict = True
            observation_dict = {key: observation_space for key,
                                observation_space in observation_spaces}
            self.get_logger().info(
                f"Observation space is of type Dict and"
                f" has the following description:")
            for name, subspace in observation_spaces:
                self.get_logger().info(f"{name} has shape {subspace.shape}")
            return spaces.Dict(observation_dict)
        else:
            # single observation space necessary
            self.get_logger().info(
                f"Observation space is NOT of type Dict and has the following"
                " description: {observation_spaces[0][1].shape}")
            # no dict space is needed so only the base observation space is
            # returned without a name
            return observation_spaces[0][1]

    def _get_spline_observations(self) -> List[float]:
        """Collects all observations that are part of the spline.

        Returns:
            List[float]: Observation vector containing only the spline values.
        """
        return np.array(
            [variable.current_position for variable in self._actions]
        )

    def _apply_FFD(self, path: Optional[str] = None) -> None:
        """
        Apply the Free Form Deformation using the current spline to the mesh
         and export the resulting mesh to the path given.

        Args:
            path (Optional[str]): Path to where the deformed mesh should be
                                  exported to. If None try to get path from
                                  mesh definition.
        """
        self._FFD.set_deformed_spline(self.spline.get_spline())
        self._FFD.deform_mesh()
        self.export_mesh(
            path if path is not None else self.mesh.get_export_path())

    def apply_action(self, action: Union[List[float], int]) -> None:
        """Function that applies a given action to the Spline.

        Args:
            action ([type]):  Action value depends on if the ActionSpace is
                              discrete (int - Signifier of the action) or
                              Continuous (List[float] - Value for each
                              continuous variable.)
        """
        # self.get_logger().debug(f"Applying action {action}")
        if self.discrete_actions:
            increasing: bool = (action % 2 == 0)
            action_index: int = int(action/2)
            self.get_logger().debug(
                f"Setting discrete action, of variable {action_index}.")
            self._actions[action_index].apply_discrete_action(increasing)
            # TODO check if this is correct
        else:
            action: List[int]
            for new_value, action_obj in zip(action, self._actions):
                action_obj.apply_continuos_action(new_value)

    def is_multiprocessing(self) -> int:
        """
        Function checks if the environment is setup to be used with
        multiprocessing Solver. Returns the number of cores the solver should
        use. If no multiprocessing 1 core is returned.

        Returns:
            int: Number of cores used in multiprocessing. 1 If no
                 multiprocessing. (Single thread still ok.)
        """
        if self.multi_processing is None:
            return 1
        return self.multi_processing.number_of_cores

    def get_validation_id(self) -> Optional[int]:
        """
        Checks if current environment has validation values if return the
        correct one otherwise return None.

        Returns:
            Optional[int]: Check text above.
        """
        if self._validation_ids is not None:
            # if self._validation_iteration >= len(self._validation_ids):
            #     before = self._validation_iteration
            #     self._validation_iteration = 0
            #     self.get_logger().warning(f"Resetting the validation "
            #                                "iteration of value {before} to "
            #                                "the first validation value 0. If"
            #                                " this happens please investigate"
            #                                " why this happens.")
            return self._validation_ids[
                self._current_validation_idx % len(self._validation_ids)
            ]
        return None

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """
        Function that is called for each step. Contains all steps that are
        performed during each step inside the environment.

        Args:
            action (Any): Action value depends on if the ActionSpace is
                          discrete (int - Signifier of the action) or
                          Continuous (List[float] - Value for each continuous
                          variable.)

        Returns:
            Tuple[Any, float, bool, Dict[str, Any]]: [description]
        """
        start = timer()
        # apply new action
        self.get_logger().info(f"Action {action}")
        self.apply_action(action)

        # apply Free Form Deformation
        self._apply_FFD()

        observations = {}
        done = False
        reward = 0.
        info = {
            "mesh_path": str(self.mesh.mxyz_path)
        }

        # run solver
        observations, reward, done, info = self.spor.run(step_information=(
            observations, done, reward, info),
            validation_id=self.get_validation_id(),
            core_count=self.is_multiprocessing(), reset=False,
            environment_id=self._id)

        # check if spline has not changed. But only in validation phase to exit
        # episodes that are always repeating the same action without breaking.
        if not done:
            self._timesteps_in_episode += 1
            if self.max_timesteps_in_episode and \
               self.max_timesteps_in_episode > 0 and \
               self._timesteps_in_episode >= self.max_timesteps_in_episode:
                done = True
                reward += self.reward_on_episode_exceeds_max_timesteps
                info["reset_reason"] = "max_timesteps"
            # self.get_logger().info(f"Checked if max timestep was reached: "
            # "{self._max_timesteps_in_episode > 0 and "
            # "self._timesteps_in_episode > self._max_timesteps_in_episode}.")
            if self.end_episode_on_spline_not_changed and \
               self._last_observation is not None:
                if np.allclose(np.array(self._last_observation),
                               np.array(observations)
                               ):  # check
                    self.get_logger().info("The Spline observation have"
                                           " not changed will exit episode.")
                    reward += self.reward_on_spline_not_changed
                    done = True
                    info["reset_reason"] = "SplineNotChanged"
            self._last_observation = copy(observations)
        else:
            if reward >= 5.:
                if self.save_good_episode_results:
                    for elem in self._result_values_to_save:
                        if np.isclose(
                            list(observations.values())[0][1],
                            elem, atol=0.05
                        ):
                            self.save_current_solution_as_png(
                                self.save_location/"episode_end_results"
                                ""/str(elem)/f""
                                f"{list(observations.values())[0][1]}"
                                f"_{self._approximate_episode}.png",
                                height=2, width=2, dpi=400)
                            break
                if self.save_random_good_episode_results:
                    if self._number_exported < 200:
                        if np.random.default_rng().random() < 5:
                            self.save_current_solution_as_png(
                                self.save_location/"episode_end_results" /
                                f"{self._approximate_episode}.png",
                                height=2, width=2, dpi=400)
                            if self.save_random_good_episode_mesh:
                                self.export_mesh(
                                    self.save_location/"episode_end_results" /
                                    "mesh"/f"{self._approximate_episode}.xns")
                            self._number_exported += 1

        self.get_logger().info(
            f"Current reward {reward} and episode is done: {done}.")

        self._last_step_results = {
            "observations": observations,
            "reward": reward,
            "done": done,
            "info": info
        }
        self.get_logger().debug(self._last_step_results)
        if self._validation_ids and self._save_image_in_validation:
            self._validation_timestep += 1
            self.save_current_solution_as_png(
                self.save_location/"validation"
                / str(self._validation_iteration)
                / str(self._current_validation_idx)
                / f"{self._validation_timestep}.png")

        if self.use_cnn_observations:
            # Need to transpose because torch wants
            # channel first representation
            observations["base_observation"] = self.get_visual_representation(
                sol_len=3).T
        else:
            observations["base_observation"] = self._get_spline_observations()
        end = timer()
        self.get_logger().debug(f"Step took {end-start} seconds.")

        return self.check_observations(observations), reward, done, info

    def check_observations(
            self, observations: ObservationType) -> ObservationType:
        self.get_logger().debug(
            f"The observations are as follows: {observations}")
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

    def reset(self) -> ObservationType:
        """
        Function that is called when the agents wants to reset the environment.
         This can either be the case if the episode is done due to #time_steps
         or the environment emits the done signal.

        Returns:
            Tuple[Any]: Observation of the newly reset environment.
        """
        self.get_logger().info("Resetting the Environment.")
        # reset spline

        # reset spline with new positions of the spline control points
        if self.reset_with_random_control_points:
            self.apply_random_action(self.get_validation_id())
        # reset the control points of the spline to the original position
        # (positions defined in the json file)
        else:
            self.spline.reset()

        # apply Free Form Deformation should now just recreate the
        # non deformed mesh
        self._apply_FFD()
        observations = {}

        done = False
        reward = 0
        info = {
            "mesh_path": str(self.mesh.mxyz_path)
        }

        self._timesteps_in_episode = 0
        self._approximate_episode += 1

        # run solver and reset reward and get new solver observations
        observations, reward, info, done = self.spor.run(
            step_information=(observations, done, reward, info),
            validation_id=self.get_validation_id(),
            core_count=self.is_multiprocessing(),
            reset=True, environment_id=self._id)

        # obs = self._get_observations(observations, done=done)

        if self._validation_ids:
            # export mesh at end of validation
            if self._current_validation_idx > 0 and \
                    self._validation_base_mesh_path:
                # validation is performed in single environment
                # with no multi threading so this is not necessary.
                base_path = pathlib.Path(
                    self._validation_base_mesh_path).parents[0]/str(
                    self._validation_iteration)/str(
                    self._current_validation_idx)
                file_name = pathlib.Path(self._validation_base_mesh_path).name
                if "_." in self._validation_base_mesh_path:
                    validation_mesh_path = base_path / str(file_name).replace(
                        "_.", f"{self._get_reset_reason_string()}.")
                else:
                    validation_mesh_path = self._validation_base_mesh_path
                self.export_mesh(validation_mesh_path)
                self.export_spline(validation_mesh_path.with_suffix(".xml"))
            if self._current_validation_idx >= len(self._validation_ids):
                self.get_logger().info(
                    "The validation callback resets the environment one time "
                    "to often. Next goal state will again be the correct one.")
            self._current_validation_idx += 1
            if self._current_validation_idx > len(self._validation_ids):
                self._current_validation_idx = 0
                self._validation_iteration += 1
        self.get_logger().info("Resetting the Environment DONE.")
        if self._validation_ids and self._save_image_in_validation:
            self._validation_timestep = 0
            self.save_current_solution_as_png(
                self.save_location/"validation" /
                str(self._validation_iteration) /
                str(self._current_validation_idx) /
                f"{self._validation_timestep}.png")
        if self.use_cnn_observations:
            # Need to transpose bc torch wants channel first representation
            observations["base_observation"] = self.get_visual_representation(
                sol_len=3).T
        else:
            observations["base_observation"] = self._get_spline_observations()

        return self.check_observations(observations)

    def _get_reset_reason_string(self) -> str:
        """
        Helper function due to line limit. Only used once but could not figure
        out how to break up line. Without making it a function. Before it was
        and inline if statement.

        Returns:
            str: Reset reason if it exists else empty string
        """
        if not self._last_step_results['info'].get('reset_reason'):
            return ""
        else:
            return str(self._last_step_results['info'].get('reset_reason'))+'_'

    def set_validation(
            self, validation_values: List[float],
            base_mesh_path: Optional[str] = None,
            end_episode_on_spline_not_change: bool = False,
            max_timesteps_in_episode: int = 0,
            reward_on_spline_not_changed: Optional[float] = None,
            reward_on_episode_exceeds_max_timesteps: Optional[float] = None,
            save_image_in_validation: Optional[bool] = False):
        """
        Converts the environment to a validation environment. This environment
        now only sets the goal states to the predefined values.

        Args:
            validation_values (List[float]): List of predefined goal states.
            base_mesh_path (Optional[str], optional): [description].
                Defaults to None.
            end_episode_on_spline_not_change (bool, optional): [description].
                Defaults to False.
            max_timesteps_per_episode (int, optional): [description].
                Defaults to 0.
        """
        self._validation_ids = validation_values
        self._current_validation_idx = 0
        self._validation_base_mesh_path = base_mesh_path
        self.max_timesteps_in_episode = max_timesteps_in_episode
        self.end_episode_on_spline_not_changed = \
            end_episode_on_spline_not_change
        self.reward_on_spline_not_changed = reward_on_spline_not_changed
        self.reward_on_episode_exceeds_max_timesteps = \
            reward_on_episode_exceeds_max_timesteps
        self._save_image_in_validation = save_image_in_validation
        self.get_logger().info(
            f"Setting environment to validation. "
            f"max_timesteps {self.max_timesteps_in_episode}, "
            f"spine_not_changed {self.end_episode_on_spline_not_changed}")

    def get_gym_environment(
        self,
        logging_information: Optional[
            Dict[str, Union[str, pathlib.Path, VerbosityLevel]]] = None
    ) -> gym.Env:
        """
        Creates and configures the gym environment so it can be used
        for training.

        Returns:
            gym.Env: OpenAI gym environment that can be used to train with
                stable_baselines[3] agents.
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
                "and not before.")

        self._actions = self.spline.get_actions()
        self._FFD.set_mesh(self.mesh.get_mesh())
        self.mesh.adapt_export_path(self._id)
        env = GymEnvironment(self._set_up_actions(),
                             self._define_observation_space())
        env.step = self.step
        env.reset = self.reset
        env.close = self.close
        return Monitor(env)

    def _set_up_logger(self, logger_name: str, log_file_location: pathlib.Path,
                       logging_level: VerbosityLevel):
        """
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
            logger_name, log_file_location, logging_level, logger=logger)
        self.set_logger_name_recursively(resulting_logger.name)

    def export_spline(self, file_name: str) -> None:
        """
        Export the current spline to the given path. The export will be done
        via gustav.

        Note:
            Gustav often uses the extension to determine the format and the
            sub files of the export so be careful how you input the file path.

        Args:
            file_name (str): [description]
        """
        self._FFD.deformed_spline.export(file_name)

    def export_mesh(self, file_name: str, space_time: bool = False) -> None:
        """
        Export the current deformed mesh to the given path. The export will be
        done via gustav.

        Note:
            Gustav often uses the extension to determine the format and the
            sub files of the export so be careful how you input the file path.

        Args:
            file_name (str):
                Path to where and how gustav should export the mesh.
            space_time (bool):
                Whether or not to use space time during the
                export. Currently during the import it is assumed no space
                time mesh is given.
        """
        self._FFD.deformed_mesh.export(file_name, space_time=space_time)

    def close(self):
        """Function is called when training is stopped.
        """
        pass

    def apply_random_action(self, seed: Optional[str] = None):
        """
        Applying a random continuous action to all movable control point
        variables. Can be activated to be used during the reset of an
        environment.

        Args:
            seed (Optional[str], optional): Seed for the generation of the
                random action. The same seed will result in always the same
                action. This functionality is chosen to make validation
                possible. If None (default) a random seed will be used and
                the action will be different each time. Defaults to None.
        """
        def _parse_string_to_int(string: str) -> int:
            chars_as_ints = [ord(char) for char in str(string)]
            string_as_int = sum(chars_as_ints)
            return string_as_int
        seed = _parse_string_to_int(str(seed)) if seed is not None else seed
        self.get_logger().debug(
            f"A random action is applied during reset with the following "
            f"seed {str(seed)}")
        rng_gen = np.random.default_rng(seed)
        random_action = (rng_gen.random((len(self._actions),))*2)-1
        for new_value, action_obj in zip(random_action, self._actions):
            action_obj.apply_continuos_action(new_value)
        return random_action

    def get_visual_representation(self, /, *, sol_len: int = 3,
                                  height: int = 10, width: int = 10,
                                  dpi: int = 20):
        """
        Returns an array representing the calculated solution. This function
        is used to create the array which is used if the solution/cnn
        representation of the base observations are selected.

        Note:
            The calculated solution can only be found if the
            :py:class:`SporObject` of the solver is called main_solver and uses
            the xns solver.
        #TODO make it broader in its application e.g. other solvers?

        Args:
            sol_len (int, optional): Number of variables to return per data
                point, input can be more but not less. Assumed is u,v,p.
                Defaults to 3.
            height (int, optional): Height of the image array in inches.
                Defaults to 10.
            width (int, optional): Width of the image array in inches.
                Defaults to 10.
            dpi (int, optional): DPI of the image array. Together with the
                height and width, this variable defines the shape of the
                return array shape=(height*dpi, width*dpi, sol_len).
                Defaults to 20.

        Raises:
            RuntimeError: could not find solution file, mesh is of the
                incorrect type (needs to be triangular), mesh is of the
                incorrect dimensions (needs to be dim=2)
        """
        # Loading the correct data and checking if correct attributes are set.
        if self._connectivity is None:
            self._connectivity = read_mixd_double(
                self.mesh.get_export_path().parent/"mien",
                3, 4, ">i").astype(int)-1
        solution_location = next(
            (x for x in self.spor.steps if x.name == "main_solver"), None)
        if solution_location:
            solution_location = pathlib.Path(
                solution_location.working_directory).expanduser().resolve()
        else:
            raise RuntimeError(
                "Could not find the main_solver spor step. Please check if "
                "you actually want to use this function, as it only works "
                "with the XNS solver.")
        if self.mesh.hypercube:
            raise RuntimeError(
                "The given mesh is not a mesh with triangular entities. "
                "This function was designed to work with only those.")
        if not self.mesh.dimensions == 2:
            raise RuntimeError(
                "The given mesh has a dimension unequal two. This function "
                "was designed to work only with meshes of dimension two.")
        solution = read_mixd_double(solution_location/"ins.out", 3)
        coordinates = self._FFD.deformed_unit_mesh_.vertices
        # Plotting and creating resulting array
        limits_max = [1, 1, 0.2e8]
        limits_min = [-1, -1, -0.2e8]

        return get_tricontour_solution(width, height, dpi, coordinates,
                                       self._connectivity, solution, sol_len,
                                       limits_min, limits_max)

    def save_current_solution_as_png(
            self, save_location: Union[pathlib.Path, str],
            include_pressure: bool = True, height: int = 10,
            width: int = 10, dpi: int = 400):
        """
        Save the current solver solution as an image at the given location.
        Additional parameters for size can be used.

        Args:
            save_location (Union[pathlib.Path, str]): Where the image is to
                be saved to. Please add solution specific path, else the file
                will be overwritten if multiple solution are saved.
            include_pressure (bool, optional): Not only use x-,y-velocity
                fields but also the pressure field. Defaults to True.
            height (int, optional): Self explanatory. Defaults to 10.
            width (int, optional): Self explanatory. Defaults to 10.
            dpi (int, optional): Self explanatory. Defaults to 400.
        """
        image_arr = self.get_visual_representation(
            sol_len=3 if include_pressure else 2,
            width=height, height=width, dpi=dpi)
        meta_data_dict = {
            "Author": "Clemens Fricke",
            "Software": "SbSOvRL",
            "Creation Time": str(datetime.datetime.now()),
            "Description": "This is a description"
        }
        if isinstance(save_location, str):
            save_location = pathlib.Path(save_location)
        save_location.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(save_location, image_arr, metadata=meta_data_dict)
