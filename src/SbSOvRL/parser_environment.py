"""Files defines the environment used for parsing and also most function which hold the functionality for the Reinforcement Learning environment are defined here.
"""
import multiprocessing, pathlib, datetime
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
from SbSOvRL.util.sbsovrl_types import ObservationType, RewardType
from SbSOvRL.util.logger import VerbosityLevel, set_up_logger
from SbSOvRL.util.load_binary import load_mixd, read_mixd_double

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
    discrete_actions: bool = True   #: Whether or not to use discret actions if False continuous actions will be used.
    reset_with_random_control_points: bool = False   #: Whether or not to reset the controlable control point variables to a random state (True) or the default original state (False). Default False.
    max_timesteps_in_episode: Optional[conint(ge=1)] = None #: maximal number of timesteps to run each episode for
    end_episode_on_spline_not_changed: bool = False #: whether or not to reset the environment if the spline has not change after a step
    reward_on_spline_not_changed: Optional[float] = None    #: reward if episode is ended due to reaching max step in episode 
    reward_on_episode_exceeds_max_timesteps: Optional[float] = None #: reward if episode is ended due to spline not changed
    use_cnn_observations: bool = False  #: use a cnn based feature extractor, if false the base observations will be supplied by the movable spline coordinates current values.

    # object variables
    _id: UUID4 = PrivateAttr(default=None)  #: id if the environment, important for multi-environment learning
    _actions: List[VariableLocation] = PrivateAttr()    #: list of actions
    _validation_ids: Optional[List[float]] = PrivateAttr(default=None)  #: if validation environment validation ids are stored here
    _current_validation_idx: Optional[int] = PrivateAttr(default=None)  #: id of the current validation id
    _timesteps_in_episode: Optional[int] = PrivateAttr(default=0)   #: number of timesteps currently spend in the episode
    _last_observation: Optional[ObservationType] = PrivateAttr(default=None)    #: last observation from the last step used to determin whether or not the spline has changed between episodes
    _FFD: FreeFormDeformation = PrivateAttr(default_factory=FreeFormDeformation)    #: FreeFormDeformation used for the spline based shape optimization
    _last_step_results: Dict[str, Any] = PrivateAttr(default={})    #: StepReturn values from last step
    _validation_base_mesh_path: Optional[str] = PrivateAttr(default=None)   #: path where the mesh should be saved to for validation
    _validation_iteration: Optional[int] = PrivateAttr(default=0)   #: How many validations were already evaluated
    _connectivity: Optional[np.ndarray] = PrivateAttr(default=None) #: The triangular connectivity of the 
    _save_image_in_validation: bool = PrivateAttr(default=False)
    _validation_timestep: int = PrivateAttr(default=0)
    _observation_is_dict: bool = PrivateAttr(default=False)


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
        observation_spaces: List[Tuple[str, gym.Space]] = []
        # define base observation
        if self.use_cnn_observations:
            observation_spaces.append(("base_observation", spaces.Box(low=0, high=255, shape=(200,200,3), dtype=np.uint8)))
        else:
            observation_spaces.append(("base_observation", spaces.Box(low=0, high=1, shape=(len(self._actions)), dtype=np.float32)))
        
        # define spor observations
        spor_obs = self.spor.get_number_of_observations()
        if spor_obs is not None:
            for item in spor_obs:
                observation_spaces.append(item.get_observation_definition())
        
        # return a dict space where each space is an entry
        if len(observation_spaces) > 1:
            self._observation_is_dict = True
            observation_dict = {key: observation_space for key, observation_space in observation_spaces}
            self.get_logger().info(f"Observation space is of type Dict and has the following description: {observation_dict}")
            return spaces.Dict(observation_dict)
        
        self.get_logger().info(f"Observation space is NOT of type Dict and has the following description: {observation_spaces}")
        return observation_spaces[0][1] # no dict space is needed so only the base observation space is returned without a name

    def _get_spline_observations(self) -> List[float]:
        """Collects all observations that are part of the spline.

        Returns:
            List[float]: Observation vector containing only the spline values.
        """
        return np.array([variable.current_position for variable in self._actions])

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
            self.get_logger().debug(f"Setting discrete action, of variable {action_index}, to {'increase'if increasing else 'decrease'} the value.")
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
        observations, reward, done, info = self.spor.run(step_information=(observations, done, reward, info) ,validation_id=self.get_validation_id(), core_count=self.is_multiprocessing(), reset=False, environment_id = self._id)

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
        self.get_logger().debug(self._last_step_results)
        if self._validation_ids and self._save_image_in_validation:
            self._validation_timestep += 1
            self.save_current_solution_as_png(self.save_location/"validation"/str(self._validation_iteration)/str(self._current_validation_idx)/f"{self._validation_timestep}.png")

        if self.use_cnn_observations:
            observations["base_observation"] = self.get_visual_representation(sol_len=3)
        else:
            observations["base_observation"] = self._get_spline_observations()
        if len(observations.keys()) == 1:
            observations = observations["base_observation"]
        end = timer()
        self.get_logger().debug(f"Step took {end-start} seconds.")

        return observations, reward, done, info

    def reset(self) -> ObservationType:
        """Function that is called when the agents wants to reset the environment. This can either be the case if the episode is done due to #time_steps or the environment emits the done signal.

        Returns:
            Tuple[Any]: Reward of the newly resetted environment.
        """
        self.get_logger().info("Resetting the Environment.")
        # reset spline
        if self.reset_with_random_control_points:  # reset spline with new positions of the spline control points
            self.apply_random_action(self.get_validation_id())
        else:   # reset the control points of the spline to the original position (positions defined in the json file)
            self.spline.reset()

        # apply Free Form Deformation should now just recreate the non deformed mesh
        self._apply_FFD()
        observations = {}

        done = False
        reward = 0
        info = {
            "mesh_path": str(self.mesh.mxyz_path)
        }

        self._timesteps_in_episode = 0


        # run solver and reset reward and get new solver observations
        observations, reward, info, done = self.spor.run(step_information=(observations, done, reward, info), validation_id=self.get_validation_id(), core_count=self.is_multiprocessing(), reset=True, environment_id = self._id)

        # obs = self._get_observations(observations, done=done)
        
        if self._validation_ids:
            # export mesh at end of validation
            if self._current_validation_idx > 0 and self._validation_base_mesh_path:
                # validation is performed in single environment with no multi threading so this is not necessary.
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
        self.get_logger().info("Resetting the Environment DONE.")
        if self._validation_ids and self._save_image_in_validation:
            self._validation_timestep = 0
            self.save_current_solution_as_png(self.save_location/"validation"/str(self._validation_iteration)/str(self._current_validation_idx)/f"{self._validation_timestep}.png")
        if self.use_cnn_observations:
            observations["base_observation"] = self.get_visual_representation(sol_len=3)
        else:
            observations["base_observation"] = self._get_spline_observations()
        if len(observations.keys()) == 1:
            observations = observations["base_observation"]
        return observations

    def set_validation(self, validation_values: List[float], base_mesh_path: Optional[str] = None, end_episode_on_spline_not_change: bool = False, max_timesteps_in_episode: int = 0, reward_on_spline_not_changed: Optional[float] = None, reward_on_episode_exceeds_max_timesteps: Optional[float] = None, save_image_in_validation: Optional[bool] = False):
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
        self._save_image_in_validation = save_image_in_validation
        self.get_logger().info(f"Setting environment to validation. max_timesteps {self.max_timesteps_in_episode}, spine_not_changed {self.end_episode_on_spline_not_changed}")

    def get_gym_environment(self, logging_information: Optional[Dict[str, Union[str, pathlib.Path, VerbosityLevel]]] = None) -> gym.Env:
        """Creates and configures the gym environment so it can be used for training.

        Returns:
            gym.Env: OpenAI gym environment that can be used to train with stable_baselines[3] agents.
        """
        if logging_information:
            self._set_up_logger(**logging_information)
        self.get_logger().info("Setting up Gym environment.")
        if self._id is None:
            self._id = uuid4()
        else:
            self.get_logger().warning(f"During setup of gym environment: The id of the environment is already set to {self._id}. Please note that when using multi environment training this will lead to errors. Please use this function only after multiplying the environments and not before.")
        
        self._actions = self.spline.get_actions()
        self._FFD.set_mesh(self.mesh.get_mesh())
        self.mesh.adapt_export_path(self._id)
        env = GymEnvironment(self._set_up_actions(),
                             self._define_observation_space()) 
        env.step = self.step
        env.reset = self.reset
        return Monitor(env)

    def _set_up_logger(self, logger_name: str, log_file_location: pathlib.Path, logging_level: VerbosityLevel):
        logger = multiprocessing.get_logger()
        resulting_logger = set_up_logger(logger_name, log_file_location, logging_level, logger=logger)
        self.set_logger_name_recursively(resulting_logger.name)

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
        
    def close(self):
        """Function is called when training is stopped.
        """
        pass


    def apply_random_action(self, seed:Optional[str] = None):
        """Applying a random continuous action to all movable control point variables. Can be activated to be used during the reset of an environment.

        Args:
            seed (Optional[str], optional): Seed for the generation of the random action. The same seed will result in always the same action. This functionality is chosen to make validation possible. If None (default) a random seed will be used and the action will be different each time. Defaults to None.
        """
        def _parse_string_to_int(string: str) -> int:
            chars_as_ints = [ord(char) for char in str(string)]
            string_as_int = sum(chars_as_ints)
            return string_as_int
        seed = _parse_string_to_int(str(seed)) if seed is not None else seed
        self.get_logger().debug(f"A random action is applied during reset with the following seed {str(seed)}")
        rng_gen = np.random.default_rng(seed)
        random_action = (rng_gen.random((len(self._actions),))*2)-1
        for new_value, action_obj in zip(random_action, self._actions):
            action_obj.apply_continuos_action(new_value)
        return random_action
    
    def get_visual_representation(self, /, *, sol_len: int=3, height: int = 10, width: int = 10, dpi: int = 20):
        """_summary_

        Args:
            sol_len (int, optional): _description_. Defaults to 3.
            height (int, optional): _description_. Defaults to 10.
            width (int, optional): _description_. Defaults to 10.
            dpi (int, optional): _description_. Defaults to 20.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
            RuntimeError: _description_

        Returns:
            _type_: _description_
        """
        # Loading the correct data and checking if correct attributes are set.
        if self._connectivity is None:
            self._connectivity = read_mixd_double(self.mesh.get_export_path().parent/"mien", 3, 4, ">i").astype(int)-1
        solution_location = next((x for x in self.spor.steps if x.name == "main_solver"), None)
        if solution_location:
            solution_location = pathlib.Path(solution_location.working_directory).expanduser().resolve()
        else: 
            raise RuntimeError("Could not find the main_solver spor step. Please check if you actually want to use this function, as it only works with the XNS solver.")
        if self.mesh.hypercube:
            raise RuntimeError("The given mesh is not a mesh with triangular entities. This function was designed to work with only those.")
        if not self.mesh.dimensions == 2:
            raise RuntimeError("The given mesh has a dimension unequal two. This function was designed to work only with meshes of dimension two.")
        solution = read_mixd_double(solution_location/"ins.out", 3)
        coordinates = self._FFD.deformed_unit_mesh_.vertices
        # Plotting and creating resulting array
        limits_max = [1,1,0.2e8]
        limits_min = [-1,-1,-0.2e8]
        arrays = []
        for i in range(sol_len):
            fig = plt.figure(figsize=(width,height), dpi=dpi)
            if i == 2:
                mappable = plt.gca().tricontourf(coordinates[:,0],coordinates[:,1],np.clip(solution[:,i], limits_min[i], limits_max[i])-1, triangles=self._connectivity, cmap="Greys", vmin=limits_min[i], vmax=limits_max[i])
            else:
                mappable = plt.gca().tricontourf(coordinates[:,0],coordinates[:,1],solution[:,i], triangles=self._connectivity, cmap="Greys", vmin=limits_min[i], vmax=limits_max[i])
            # plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            # plt.axis("equal")
            ax=plt.gca()
            ax.set_xlim((0,1))
            ax.set_ylim((0,1))
            # ax.use_sticky_edges = False
            # plt.colorbar(mappable)
            # plt.title("Flow Field")
            # plt.autoscale(True)
            # for col in ax.collections:
            #   print(col)
            ax.margins(tight=True)
            plt.axis('off')
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            plt.tight_layout(pad=0, w_pad=0, h_pad=0)
            # fig.canvas.draw()
            # sio = StringIO()
            # arrays.append(np.asarray(fig.canvas.get_renderer().buffer_rgba()))
            # sio.close()
            fig.canvas.draw()
            # Now we can save it to a numpy array.
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            arrays.append(data)
            plt.close()
        arr = np.array(arrays[0])
        for i in range(sol_len-1):
            arr[:,:,i+1] = arrays[i+1][:,:,0]
        return arr
    
    def save_current_solution_as_png(self, save_location: Union[pathlib.Path, str], include_pressure: bool = True):
        image_arr = self.get_visual_representation(sol_len=3 if include_pressure else 2, width=10, height=10, dpi=100)
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