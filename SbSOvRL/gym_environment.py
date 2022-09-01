"""
The gym environment is a class that incorporates the interface requirements set
out by the OpenAI gym library for Reinforcement Learning environment. This
standard is very commonly used in python based Reinforcement Learning
applications and most python libraries supplying Reinforcement Learning agents
can use this interface.

This interface was not incorporated into the SbSOvRL.parser_environment class
due to incompatibility between the pydantic and gym functionality. It is also
advantageous to separate the two classes since now it is possible to make
multiple instances of the environment accessible for multi environment training
and separate validation environment.

An instance of this class is created by the
SbSOvRL.parser_environment.Environment.get_gym_environment function. Each call
of this function creates a new gym environment.

Note:
    Currently it is not possible to run xns solvers concurrently out of the box
    since currently all created environments use the same run location. This
    has the result that the output files of the xns solver are also created in
    the same folder for all environments and therefore overwrite each other. A
    fix for this problem is currently worked on.
"""

from gym import Env
from typing import Dict, Any, Tuple, Union, List


class GymEnvironment(Env):
    """
    This class is a placeholder class which complies with the OpenAI gym
    Interface. The real functionality is infused into the class after creation
    by substituting the functions from the
    SbSOvRL.parser_environment.Environment class.
    """
    metadata = {'render.modes': ['mesh']}

    def __init__(self, action_space, observation_space) -> None:
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space

    def step(
            self,
            action: Union[int, List[float]]
            ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        pass

    def reset(self) -> Any:
        pass

    def render(self, mode: str = "mesh") -> None:
        pass

    def close(self) -> None:
        pass
