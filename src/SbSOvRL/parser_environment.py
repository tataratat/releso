"""Files defines the environment used for parsing and also most function which hold the functionality for the Reinforcement Learning environment are defined here.
"""
from pydantic import conint
from typing import Optional, Any, List, Dict, Tuple, Union

from pydantic.class_validators import validator
from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.spline import Spline, VariableLocation
from SbSOvRL.util.logger import get_parser_logger
from SbSOvRL.mesh import Mesh
from SbSOvRL.spor import MultiProcessor, SPORList
from gustav import FreeFormDeformation
from pydantic.fields import Field, PrivateAttr
import gym
from gym import spaces
from SbSOvRL.gym_environment import GymEnvironment
import numpy as np
from copy import copy
from stable_baselines3.common.monitor import Monitor
from SbSOvRL.base_model import SbSOvRL_BaseModel
import pathlib

from SbSOvRL.util.sbsovrl_types import ObservationType, RewardType

class MultiProcessing(SbSOvRL_BaseModel):
    """Defines if the Problem should use Multiprocessing and with how many cores the solver can work. Does not force Multiprocessing for example if the solver does not support it.
    """
    number_of_cores: conint(ge=1) = 1   #: Maximal number of cores which can be used by the current environment. Multi-Environments will use multiple of these.


class Environment(SbSOvRL_BaseModel):
    """
    Parser Environment object is created by pydantic during the parsing of the json object defining the Spline base Shape optimization. Each object can create a gym environment that represents the given problem.
    """
    multi_processing: Optional[MultiProcessing] = Field(default_factory=MultiProcessor) #: defines if multi-processing can be used.
    spline: Spline  #: definition of the spline
    mesh: Mesh  #: definition of the mesh
    spor: SPORList  #: definition of the spor objects
    discrete_actions: bool = True   #: Whether or not to use discreet actions if False continuous actions will be used.
    max_timesteps_in_episode: Optional[conint(ge=1)] = None #: maximal number of timesteps to run each episode for
    end_episode_on_spline_not_changed: bool = False #: whether or not to reset the environment if the spline has not change after a step
    reward_on_spline_not_changed: Optional[float] = None    #: reward if episode is ended due to reaching max step in episode 
    reward_on_episode_exceeds_max_timesteps: Optional[float] = None #: reward if episode is ended due to spline not changed

    # object variables
    _id: Optional[int] = PrivateAttr(default=None)  #: id if the environment, important for multi-environment learning
    _actions: List[VariableLocation] = PrivateAttr()    #: list of actions
    _validation_ids: Optional[List[float]] = PrivateAttr(default=None)  #: if validation environment validation ids are stored here
    _current_validation_idx: Optional[int] = PrivateAttr(default=None)  #: id of the current validation id
    _timesteps_in_episode: Optional[int] = PrivateAttr(default=0)   #: number of timesteps currently spend in the episode
    _last_observation: Optional[ObservationType] = PrivateAttr(default=None)    #: last observation from the last step used to determin whether or not the spline has changed between episodes
    _FFD: FreeFormDeformation = PrivateAttr(default_factory=FreeFormDeformation)    #: FreeFormDeformation used for the spline based shape optimization
    _last_step_results: Dict[str, Any] = PrivateAttr(default={})    #: StepReturn values from last step
    _validation_base_mesh_path: Optional[str] = PrivateAttr(default=None)   #: path where the mesh should be saved to for validation
    _validation_iteration: Optional[int] = PrivateAttr(default=0)   #: How many validations were already evaluated


    @validator("reward_on_spline_not_changed", always=True)
    @classmethod
    def check_if_reward_given_if_spline_not_change_episode_killer_activated(cls, value, values) -> float:
        """Checks that 1) if a reward is set, also the boolean value for the end_episode_on_spline_not_changed is True. 2) If end_episode_on_spline_not_changed is True a reward value is set.

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values (here end_episode_on_spline_not_changed is important)

        Raises:
            SbSOvRLParserException: Error is thrown if one of the conditions is not met.

        Returns:
            float: reward for the specified occurrence.
        """
        if "end_episode_on_spline_not_changed" not in values:
            raise SbSOvRLParserException("Environment", "reward_on_spline_not_changed", "Could not find definition of parameter end_episode_on_spline_not_changed, please defines this variable since otherwise this variable would have no function.")
        if value is not None and (values["end_episode_on_spline_not_changed"] is None or not values["end_episode_on_spline_not_changed"]):
            raise SbSOvRLParserException("Environment", "reward_on_spline_not_changed", "Reward can only be set if end_episode_on_spline_not_changed is true.")
        if values["end_episode_on_spline_not_changed"] and value is None:
            get_parser_logger().warning("Please set a reward value for spline not changed if episode should end on it. Will set 0 for you now, but this might not be you intention.")
            value = 0.
        return value

    @validator("reward_on_episode_exceeds_max_timesteps", always=True)
    @classmethod
    def check_if_reward_given_if_max_steps_killer_activated(cls, value, values) -> int:
        """Checks that 1) if a reward is set, also the boolean value for the max_timesteps_in_episode is True. 2) If max_timesteps_in_episode is True a reward value is set.

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values (here max_timesteps_in_episode is important)

        Raises:
            SbSOvRLParserException: Error is thrown if one of the conditions is not met.

        Returns:
            float: reward for the specified occurrence.
        """
        if "max_timesteps_in_episode" not in values:
            raise SbSOvRLParserException("Environment", "reward_on_episode_exceeds_max_timesteps", "Could not find definition of parameter max_timesteps_in_episode, please defines this variable since otherwise this variable would have no function.")
        if value is not None and (values["max_timesteps_in_episode"] is None or not values["max_timesteps_in_episode"]):
            raise SbSOvRLParserException("Environment", "reward_on_episode_exceeds_max_timesteps", "Reward can only be set if max_timesteps_in_episode a positive integer.")
        if values["max_timesteps_in_episode"] and value is None:
            get_parser_logger().warning("Please set a reward value for max time steps exceeded, if episode should end on it. Will set 0 for you now, but this might not be you intention.")
            value = 0.
        return value
    # object functions

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._actions = self.spline.get_actions()
        self._FFD.set_mesh(self.mesh.get_mesh())

    def _set_up_actions(self) -> gym.Space:
        """Creates the action space the gym environment uses to define its action space.

        Returns:
            # gym.Space: action space of the current problem.
        """
        if self.discrete_actions:
            return spaces.Discrete(len(self._actions) * 2)
        else:
            return spaces.Box(low=-1, high=1, shape=(len(self._actions),))

    def _define_observation_space(self) -> gym.Space:
        """Creates the observation space the gym environment uses to define its observations space.

        Returns:
            # gym.Space: Observation space of the current problem.
        """
        # TODO This needs to be changed
        return spaces.Box(low=-1, high=1, shape=(len(self._actions)+self.spor.get_number_of_observations(),))

    def _get_spline_observations(self) -> List[float]:
        """Collects all observations that are part of the spline.

        Returns:
            List[float]: Observation vector containing only the spline values.
        """
        return [variable.current_position for variable in self._actions]

    # def _get_observations(self, new_observations: ObservationType, done: bool) -> List[float]:
    #     """Collects all observations (Spline observations and solver observations) into a single observation vector.

    #     Note: This tool is currently setup to only work with and MLP policy since otherwise a multidimensional observation vector needs to be created.

    #     Args:
    #         new_observations (ObservationType): Dict containing the reward and observations from the solver given by the solver.
    #         done (bool): If done no solver obseration can be read in so 0 values will be added for them.

    #     Returns:
    #         List[float]: Vector of the observations.
    #     """
    #     # get spline observations
    #     observations = self._get_spline_observations()

    #     # add solver observations
    #     if done: # if solver failed use 0 solver observations
    #         observations.extend([0 for _ in range(self.spor.get_number_of_observations())])
    #     elif new_observations is not None: # use observed observations
    #         observations.extend(
    #             [item for item in new_observations])
    #     obs = np.array(observations)
    #     return obs

    def _apply_FFD(self, path: Optional[str] = None) -> None:
        """Apply the Free Form Deformation using the current spline to the mesh and export the resulting mesh to the path given.

        Args:
            path (Optional[str]): Path to where the deformed mesh should be exported to. If None try to get path from mesh defintion.
        """
        self._FFD.set_deformed_spline(self.spline.get_spline())
        self._FFD.deform_mesh()
        self.export_mesh(path if path is not None else self.mesh.get_export_path())

    def apply_action(self, action: Union[List[float], int]) -> None:
        """Function that applies a given action to the Spline.

        Args:
            action ([type]):  Action value depends on if the ActionSpace is discrete (int - Signifier of the action) or Continuous (List[float] - Value for each continuous variable.)
        """
        # self.get_logger().debug(f"Applying action {action}")
        if self.discrete_actions:
            increasing: bool = (action%2 == 0)
            action_index: int = int(action/2)
            self._actions[action_index].apply_discrete_action(increasing)
            # TODO check if this is correct
        else:
            action: List[int]
            for new_value, action_obj in zip(action, self._actions):
                action_obj.apply_continuos_action(new_value)

    def is_multiprocessing(self) -> int:
        """Function checks if the environment is setup to be used with multiprocessing Solver. Returns the number of cores the solver should use. If no multiprocessing 1 core is returned.

        Returns:
            int: Number of cores used in multiprocessing. 1 If no multiprocessing. (Single thread still ok.)
        """
        if self.multi_processing is None:
            return 1
        return self.multi_processing.number_of_cores

    def get_validation_id(self) -> Optional[int]:
        """Checks if current environment has validation values if return the correct one otherwise return None.

        Returns:
            Optional[int]: Check text above.
        """
        if self._validation_ids is not None:
            # if self._validation_iteration >= len(self._validation_ids):
            #     before = self._validation_iteration
            #     self._validation_iteration = 0
            #     self.get_logger().warning(f"Resetting the validation iteration of value {before} to the first validation value 0. If this happens please investigate why this happens.")
            return self._validation_ids[self._current_validation_idx%len(self._validation_ids)]
        return None

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Function that is called for each step. Contains all steps that are performed during each step inside the environment.

        Args:
            action (Any): Action value depends on if the ActionSpace is discrete (int - Signifier of the action) or Continuous (List[float] - Value for each continuous variable.)

        Returns:
            Tuple[Any, float, bool, Dict[str, Any]]: [description]
        """
        # apply new action
        self.get_logger().debug(f"Action {action}")
        self.apply_action(action)

        # apply Free Form Deformation
        self._apply_FFD()

        observations = np.array(self._get_spline_observations())
        done = False
        reward = 0.
        info = {
            "mesh_path": str(self.mesh.mxyz_path)
        }

        # run solver
        observations, reward, done, info = self.spor.run((observations, done, reward, info) ,self.get_validation_id(), self.is_multiprocessing(), reset=False)

        # check if spline has not changed. But only in validation phase to exit episodes that are always repeating the same action without breaking.
        if not done:
            self._timesteps_in_episode += 1
            if self.max_timesteps_in_episode and self.max_timesteps_in_episode > 0 and self._timesteps_in_episode >= self.max_timesteps_in_episode:
                done = True
                reward += self.reward_on_episode_exceeds_max_timesteps
                info["reset_reason"] = "max_timesteps"
            # self.get_logger().info(f"Checked if max timestep was reached: {self._max_timesteps_in_episode > 0 and self._timesteps_in_episode > self._max_timesteps_in_episode}.")
            if self.end_episode_on_spline_not_changed and self._last_observation is not None:
                if np.allclose(np.array(self._last_observation), np.array(observations)): # check
                    self.get_logger().info("The Spline observation have not changed will exit episode.")
                    reward += self.reward_on_spline_not_changed
                    done = True
                    info["reset_reason"] = "SplineNotChanged" 
            self._last_observation = copy(observations)

        self.get_logger().info(f"Current reward {reward} and episode is done: {done}.")

        self._last_step_results = {
            "observations": observations,
            "reward": reward,
            "done": done,
            "info": info
        }
        return observations, reward, done, info

    def reset(self) -> ObservationType:
        """Function that is called when the agents wants to reset the environment. This can either be the case if the episode is done due to #time_steps or the environment emits the done signal.

        Returns:
            Tuple[Any]: Reward of the newly resetted environment.
        """
        self.get_logger().info("Resetting the Environment.")
        # reset spline
        self.spline.reset()

        # apply Free Form Deformation should now just recreate the non deformed mesh
        self._apply_FFD()

        observations = np.array(self._get_spline_observations())
        done = False
        reward = 0
        info = {
            "mesh_path": str(self.mesh.mxyz_path)
        }

        self._timesteps_in_episode = 0
        if self._validation_ids:
            # export mesh at end of validation
            if self._current_validation_idx > 0 and self._validation_base_mesh_path:
                # TODO check if path definition is correct NEEDS to change for multi environment learning
                base_path = pathlib.Path(self._validation_base_mesh_path).parents[0] / str(self._validation_iteration) / str(self._current_validation_idx)
                file_name = pathlib.Path(self._validation_base_mesh_path).name
                if "_." in self._validation_base_mesh_path:
                    validation_mesh_path = base_path / str(file_name).replace("_.", f"{'' if not self._last_step_results['info'].get('reset_reason') else (str(self._last_step_results['info'].get('reset_reason'))+'_')}.")
                else:
                    validation_mesh_path = self._validation_base_mesh_path
                self.export_mesh(validation_mesh_path)
                self.export_spline(validation_mesh_path.with_suffix(".xml"))
                # ffd_vis_for_rl.plot_deformed_unit_mesh(self._FFD, base_path/"deformed_unit_mesh.svg")
                # ffd_vis_for_rl.plot_deformed_spline(self._FFD, base_path/"deformed_spline.svg")
            if self._current_validation_idx >= len(self._validation_ids):
                self.get_logger().info("The validation callback resets the environment one time to often. Next goal state will again be the correct one.")
            self._current_validation_idx += 1
            if self._current_validation_idx>len(self._validation_ids):
                self._current_validation_idx = 0
                self._validation_iteration += 1


        # run solver and reset reward and get new solver observations
        observations, reward, info, done = self.spor.run((observations, done, reward, info), self.get_validation_id(), core_count=self.is_multiprocessing(), reset=True)

        # obs = self._get_observations(observations, done=done)
        return observations

    def set_validation(self, validation_values: List[float], base_mesh_path: Optional[str] = None, end_episode_on_spline_not_change: bool = False, max_timesteps_in_episode: int = 0, reward_on_spline_not_changed: Optional[float] = None, reward_on_episode_exceeds_max_timesteps: Optional[float] = None):
        """Converts the environment to a validation environment. This environment now only sets the goal states to the predefined values.

        Args:
            validation_values (List[float]): List of predefined goal states.
            base_mesh_path (Optional[str], optional): [description]. Defaults to None.
            end_episode_on_spline_not_change (bool, optional): [description]. Defaults to False.
            max_timesteps_per_episode (int, optional): [description]. Defaults to 0.
        """
        self._validation_ids = validation_values
        self._current_validation_idx = 0
        self._validation_base_mesh_path = base_mesh_path
        self.max_timesteps_in_episode = max_timesteps_in_episode
        self.end_episode_on_spline_not_changed = end_episode_on_spline_not_change
        self.reward_on_spline_not_changed = reward_on_spline_not_changed
        self.reward_on_episode_exceeds_max_timesteps = reward_on_episode_exceeds_max_timesteps
        self.get_logger().info(f"Setting environment to validation. max_timesteps {self.max_timesteps_in_episode}, spine_not_changed {self.end_episode_on_spline_not_changed}")

    def get_gym_environment(self) -> gym.Env:
        """Creates and configures the gym environment so it can be used for training.

        Returns:
            gym.Env: OpenAI gym environment that can be used to train with stable_baselines[3] agents.
        """
        self.get_logger().info("Setting up Gym environment.")
        env = GymEnvironment(self._set_up_actions(),
                             self._define_observation_space()) 
        env.step = self.step
        env.reset = self.reset
        return Monitor(env)

    def export_spline(self, file_name: str) -> None:
        """Export the current spline to the given path. The export will be done via gustav.

        Note: Gustav often uses the extension to determin the format and the sub files of the export so be careful how you input the file path.

        Args:
            file_name (str): [description]
        """
        self._FFD.deformed_spline.export(file_name)

    def export_mesh(self, file_name: str, space_time: bool = False) -> None:
        """Export the current deformed mesh to the given path. The export will be done via gustav. 
        
        Note: Gustav often uses the extension to determin the format and the sub files of the export so be careful how you input the file path.

        Args:
            file_name (str): Path to where and how gustav should export the mesh.
            space_time (bool): Whether or not to use space time during the export. Currently during the import it is assumed no space time mesh is given.
        """
        self._FFD.deformed_mesh.export(file_name, space_time=space_time)
