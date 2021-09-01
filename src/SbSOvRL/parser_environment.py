from pydantic import BaseModel, conint
from typing import Optional, Any, List, Dict, Tuple
from SbSOvRL.spline import Spline, VariableLocation
from SbSOvRL.mesh import Mesh
from SbSOvRL.solver import SolverTypeDefinition
from gustav import FreeFormDeformation
from pydantic.fields import PrivateAttr
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from SbSOvRL.gym_environment import GymEnvironment
from SbSOvRL.util.logger import set_up_logger
import numpy as np

parser_logger = set_up_logger("SbSOvRL_parser")
environment_logger = set_up_logger("SbSOvRL_environment")


class MultiProcessing(BaseModel):
    number_of_cores: conint(ge=1)


class Environment(BaseModel):
    multi_processing: Optional[MultiProcessing] = None
    spline: Spline
    mesh: Mesh
    solver: SolverTypeDefinition
    discrete_actions: bool = True
    additional_observations: conint(ge=0)

    # object variables
    _actions: List[VariableLocation] = PrivateAttr()
    _FFD: FreeFormDeformation = PrivateAttr(
        default_factory=FreeFormDeformation)

    # object functions

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._actions = self.spline.get_actions()
        self._FFD.set_mesh(self.mesh.get_mesh())

    def _set_up_actions(self) -> gym.Space:
        if self.discrete_actions:
            return spaces.Discrete(len(self._actions) * 2)
        else:
            return spaces.Box(low=-1, high=1, shape=(len(self._actions),))

    def _define_observation_space(self) -> gym.Space:
        return spaces.Box(low=-1, high=1, shape=(len(self._actions)+self.additional_observations,))

    def _get_spline_observations(self) -> List[float]:
        return [variable.current_position for variable in self._actions]

    def _get_observations(self, reward_solver_output: Dict[str, Any], done: bool) -> List[float]:
        # get spline observations
        observations = self._get_spline_observations()

        # add solver observations
        if done: # if solver failed use 0 solver observations
            observations.extend([0 for _ in range(self.additional_observations)])
        else: # use observed observations
            observations.extend(
                [item for item in reward_solver_output["observations"]])

        return np.array(observations)

    def _apply_FFD(self, path: str) -> None:
        self._FFD.set_deformed_spline(self.spline.get_spline())
        self._FFD.deform_mesh()
        self.export_mesh(path)

    def apply_action(self, action) -> None:
        print("New action", action)

    def is_multiprocessing(self):
        if self.multi_processing is None:
            return 0
        return self.multi_processing.number_of_cores

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        # apply new action
        self.apply_action(action)

        # apply Free Form Deformation
        self._apply_FFD(self.solver.working_directory /
                        "mesh/deformed/mesh.xns")
        # run solver
        reward_solver_output, done = self.solver.start_solver(
            core_count=self.is_multiprocessing())

        info = {}

        return self._get_observations(reward_solver_output=reward_solver_output, done=done), reward_solver_output["reward"], done, info

    def reset(self) -> Tuple[Any]:
        environment_logger.info("Resetting the Environment.")
        # reset spline
        self.spline.reset()

        # run solver and reset reward and get new solver observations
        reward_solver_output, done = self.solver.start_solver(
            reset=True, core_count=self.is_multiprocessing())

        return self._get_observations(reward_solver_output=reward_solver_output, done=done)

    def get_gym_environment(self) -> gym.Env:
        env = GymEnvironment(self._set_up_actions(),
                             self._define_observation_space())
        env.step = self.step
        env.reset = self.reset
        check_env(env)
        return env

    def export_spline(self, file_name: str) -> None:
        self._FFD.deformed_spline.export(file_name)

    def export_mesh(self, file_name: str) -> None:
        self._FFD.deformed_mesh.export(file_name)
