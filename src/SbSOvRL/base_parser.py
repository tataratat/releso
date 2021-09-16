from SbSOvRL.agent import AgentTypeDefinition
from SbSOvRL.parser_environment import Environment
from pydantic import BaseModel, validator
from typing import Union, Optional
from pydantic.fields import Field, PrivateAttr
from pydantic.types import conint
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, EvalCallback
from stable_baselines3.common.base_class import BaseAlgorithm
import pathlib
import datetime
from SbSOvRL.verbosity import Verbosity

class BaseParser(BaseModel):
    verbosity: Verbosity = Field(default_factory=Verbosity)
    agent: AgentTypeDefinition
    environment: Union[Environment]
    number_of_timesteps: conint(ge=1)
    number_of_episodes: conint(ge=1)
    model_save_location: Optional[str]


    # internal objects
    _agent: Optional[BaseAlgorithm] = PrivateAttr(default=None)

    @validator("model_save_location")
    @classmethod
    def add_datetime_to_model_save_location(cls, v) -> str:
        if '{}' in v:
            v = v.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        return v
        

    def learn(self) -> None:
        """
        Starts the training that is specified in the loaded json file.
        """
        self._agent = self.agent.get_agent(self.environment.get_gym_environment())

        callbacks = [
            StopTrainingOnMaxEpisodes(max_episodes=self.number_of_episodes)
            # EvalCallback(
            #     eval_env=self.environment.get_gym_enviroment(),
            #     log_path="EvalCallbackLogPath/",
            #     best_model_save_path="model_save",
            #     eval_freq=10000,
            #     verbose=1)
        ]

        

        self._agent.learn(self.number_of_timesteps, callback=callbacks, tb_log_name=self.agent.get_next_tensorboard_experiment_name())
        self.save_model()

    def export_spline(self, file_name: str) -> None:
        """Exports the current spline to the specified location.

        Args:
            file_name (str): Path to the location where the spline should be stored at.
        """
        self.environment.export_spline(file_name)

    def export_mesh(self, file_name:str) -> None:
        """Exports the current mesh to the specified location.

        Args:
            file_name (str): Path to the location where the mesh should be stored at.
        """
        self.environment.export_mesh(file_name)

    def save_model(self, file_name: Optional[str] = None) -> None:
        """Saves the current agent to the specified location or to a default location.

        Args:
            file_name (Optional[str]): Path where the agent is saved to. If None will see if json had a save location if also not given will use a default location. Defaults to None.
        """
        if file_name is None and self.model_save_location is not None:
            path = pathlib.Path(self.model_save_location)
        elif file_name is not None:
            path = pathlib.Path(file_name)
        else:
            path = pathlib.Path(f"default_model_safe_location/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._agent.save(path)
            

