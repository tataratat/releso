import gym
from gym import spaces
from typing import Dict, Any, Tuple


class GymEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, action_space, observation_space) -> None:
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
        pass

    def reset(self) -> Any:
        pass

    def render(self, mode: str = "console") -> None:
        pass

    def close(self) -> None:
        pass
