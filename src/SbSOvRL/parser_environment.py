from pydantic import conint
from typing import Optional, Any, List, Dict, Tuple, Union
from SbSOvRL.spline import Spline, VariableLocation
from SbSOvRL.mesh import Mesh
from SbSOvRL.solver import SolverTypeDefinition
from gustav import FreeFormDeformation
from pydantic.fields import PrivateAttr
import gym
from gym import spaces
from SbSOvRL.gym_environment import GymEnvironment
import logging
import numpy as np
from copy import copy
from stable_baselines3.common.monitor import Monitor
from SbSOvRL.base_model import SbSOvRL_BaseModel

class MultiProcessing(SbSOvRL_BaseModel):
    """Defines if the Problem should use Multiprocessing and with how many cores the solver can work. Does not force Multiprocessing for example if the solver does not support it.
    """
    number_of_cores: conint(ge=1)


class Environment(SbSOvRL_BaseModel):
    """
    Parser Environment object is created by pydantic during the parsing of the json object defining the Spline base Shape optimization. Each object can create a gym environment that represents the given problem.
    """
    multi_processing: Optional[MultiProcessing]
    spline: Spline
    mesh: Mesh
    solver: SolverTypeDefinition
    discrete_actions: bool = True
    additional_observations: conint(ge=0)

    # object variables
    _actions: List[VariableLocation] = PrivateAttr()
    _validation: Optional[List[float]] = PrivateAttr(default=None)
    _validation_idx: Optional[int] = PrivateAttr(default=0)
    _last_observation: np.ndarray = PrivateAttr()
    _logger_name: str = PrivateAttr(default="SbSOvRL_environment")
    _FFD: FreeFormDeformation = PrivateAttr(
        default_factory=FreeFormDeformation)

    # object functions

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._actions = self.spline.get_actions()
        self._FFD.set_mesh(self.mesh.get_mesh())
        self._last_observation = None

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
        return spaces.Box(low=-1, high=1, shape=(len(self._actions)+self.additional_observations,))

    def _get_spline_observations(self) -> List[float]:
        """Collects all observations that are part of the spline.

        Returns:
            List[float]: Observation vector containing only the spline values.
        """
        return [variable.current_position for variable in self._actions]

    def _get_observations(self, reward_solver_output: Dict[str, Any], done: bool) -> List[float]:
        """Collects all observations (Spline observations and solver observations) into a single observation vector.

        Note: This tool is currently setup to only work with and MLP policy since otherwise a multidimensional observation vector needs to be created.

        Args:
            reward_solver_output (Dict[str, Any]): Dict containing the reward and observations from the solver given by the solver.
            done (bool): If done no solver obseration can be read in so 0 values will be added for them.

        Returns:
            List[float]: Vector of the observations.
        """
        # get spline observations
        observations = self._get_spline_observations()

        # add solver observations
        if done: # if solver failed use 0 solver observations
            observations.extend([0 for _ in range(self.additional_observations)])
        else: # use observed observations
            observations.extend(
                [item for item in reward_solver_output["observations"]])
        obs = np.array(observations)
        return obs

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
        # logging.getLogger(self._logger_name).debug(f"Applying action {action}")
        if self.discrete_actions:
            increasing = (action%2 == 0)
            action_index = int(action/2)
            self._actions[action_index].apply_discrete_action(increasing)
            # TODO check if this is correct
        else:
            for new_value, action in zip(action, self._actions):
                action.apply_continuos_action(new_value)

    def is_multiprocessing(self) -> int:
        """Function checks if the environment is setup to be used with multiprocessing Solver. Returns the number of cores the solver should use. If no multiprocessing 0 cores are returned.

        Returns:
            int: Number of cores used in multiprocessing. 0 If no multiprocessing. (Single thread still ok.)
        """
        if self.multi_processing is None:
            return 0
        return self.multi_processing.number_of_cores

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Function that is called for each step. Contains all steps that are performed during each step inside the environment.

        Args:
            action (Any): Action value depends on if the ActionSpace is discrete (int - Signifier of the action) or Continuous (List[float] - Value for each continuous variable.)

        Returns:
            Tuple[Any, float, bool, Dict[str, Any]]: [description]
        """
        # apply new action
        logging.getLogger(self._logger_name).debug(f"Action {action}")
        self.apply_action(action)

        # apply Free Form Deformation
        self._apply_FFD()
        # run solver
        reward_solver_output, done = self.solver.start_solver(
            core_count=self.is_multiprocessing())

        info = {}

        observations = self._get_observations(reward_solver_output=reward_solver_output, done=done)

        # check if spline has not changed. But only in validation phase to exit episodes that are always repeating the same action without breaking.
        if self._validation:
            if done or self._last_observation is None:
                # no need to compare during reset, done or during first step
                pass
            else:
                print(self._last_observation, observations)
                print(type(self._last_observation), type(observations))
                if np.allclose(np.array(self._last_observation), np.array(observations)):
                    logging.getLogger(self._logger_name).info("The Spline observation have not changed will exit episode.")
                    done = True
            self._last_observation = copy(observations)

        logging.getLogger(self._logger_name).info(f"Current reward {reward_solver_output['reward']} and episode is done: {done}.")
        return observations, reward_solver_output["reward"], done, info

    def reset(self) -> Tuple[Any]:
        """Function that is called when the agents wants to reset the environment. This can either be the case if the episode is done due to #time_steps or the environment emits the done signal.

        Returns:
            Tuple[Any]: Reward of the newly resetted environment.
        """
        logging.getLogger(self._logger_name).info("Resetting the Environment.")
        # reset spline
        self.spline.reset()

        # apply Free Form Deformation should now just recreate the non deformed mesh
        self._apply_FFD()

        new_goal_value = None

        if self._validation:
            if self._validation_idx == len(self._validation):
                logging.getLogger(self._logger_name).info("The validation callback resets the environment one time to often. Next goal state will again be the correct one.")
            new_goal_value = self._validation[self._validation_idx%len(self._validation)]
            self._validation_idx += 1
            if self._validation_idx>len(self._validation):
                self._validation_idx = 0

        # run solver and reset reward and get new solver observations
        reward_solver_output, done = self.solver.start_solver(
            reset=True, core_count=self.is_multiprocessing(), new_goal_value=new_goal_value)

        obs = self._get_observations(reward_solver_output=reward_solver_output, done=done)
        return obs

    def set_validation(self, validation_values: List[float]):
        """Converts the environment to a validation environment. This environment now only sets the goal states to the predifined values.

        Args:
            validation_values (List[float]): List of predifiened goal states.
        """
        self._validation = validation_values
        self._logger_name = "SbSOvRL_validation_environment"


    def get_gym_environment(self) -> gym.Env:
        """Creates and configures the gym environment so it can be used for training.

        Returns:
            gym.Env: openai gym environment that can be used to train with [stable_]baselines[3] agents.
        """
        logging.getLogger(self._logger_name).info("Setting up Gym environment.")
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

    def export_mesh(self, file_name: str, space_time: bool = True) -> None:
        """Export the current deformed mesh to the given path. The export will be done via gustav. 
        
        Note: Gustav often uses the extension to determin the format and the sub files of the export so be careful how you input the file path.

        Args:
            file_name (str): Path to where and how gustav should export the mesh.
            space_time (bool): Whether or not to use space time during the export. Currently during the import it is assumed no space time mesh is given.
        """
        self._FFD.deformed_mesh.export(file_name, space_time=space_time)
