"""
File defines the base json object which is needed to define the problem setting for the command line based usage of the SbSOvRL toolbox/framework.
"""
from copy import deepcopy
from SbSOvRL.agent import AgentTypeDefinition
from SbSOvRL.parser_environment import Environment
from SbSOvRL.validation import Validation
from SbSOvRL.callback import EpisodeLogCallback
from SbSOvRL.verbosity import Verbosity
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.exceptions import SbSOvRLValidationNotSet
from SbSOvRL.parser_environment import Environment
from typing import Union, Optional, Any
from pydantic.fields import Field, PrivateAttr
from pydantic.types import conint
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.base_class import BaseAlgorithm
import pathlib
import datetime
import numpy as np


class BaseParser(SbSOvRL_BaseModel):
    """
    This class can be used to initialize the SbSOvRL Framework from the command line by reading in a json representation of the Spline based Shape Optimization problem which is to be solved via Reinforcement Learning.
    """

    verbosity: Verbosity = Field(
        default_factory=Verbosity
    )  #: Defining the verbosity of the training process and environment loading
    agent: AgentTypeDefinition  #: Definition of the agent to be used during training and/or validation of the RL use case
    environment: Environment  #: Definition of the environment which encodes the parameters of the RL use case
    number_of_timesteps: conint(
        ge=1
    )  #: Number of timesteps the training process should run for (supperseeded by the SbSOvRL.base_parser.BaseParser.number_of_episodes)
    number_of_episodes: conint(
        ge=1
    )  #: Number of episodes the training prcess should run for
    validation: Optional[Validation]  #: Definition of the validation parameters

    # internal objects
    _agent: Optional[BaseAlgorithm] = PrivateAttr(
        default=None
    )  #: Holds the trainable agent for the RL use case. The SbSOvRL.base_parser.BaseParser.agent defines the type and parameters of the agent this is the actual trainable agent.

    def __init__(__pydantic_self__, **data: Any) -> None:
        """Initializes the class correctly and also adds the correct logger to all subclasses.
        """
        super().__init__(**data)
        __pydantic_self__.set_logger_name_recursively(
            __pydantic_self__.verbosity._environment_logger
        )

    def learn(self) -> None:
        """
        Starts the training that is specified in the loaded json file.
        """
        self._agent = self.agent.get_agent(self.environment.get_gym_environment())

        callbacks = [
            StopTrainingOnMaxEpisodes(max_episodes=self.number_of_episodes),
            EpisodeLogCallback(
                episode_log_location=self.save_location / "episode_log.csv", verbose=1
            ),
        ]
        validation_environment = self._create_validation_environment()

        if self.validation:
            if self.validation.should_add_callback():
                callbacks.append(
                    self.validation.get_callback(
                        validation_environment.get_gym_environment(),
                        save_location=self.save_location,
                    )
                )

        self.get_logger().info(
            f"The environment is now trained for {self.number_of_timesteps} episodes. If the maximum number of episode callback is set this value might not mean much."
        )
        self._agent.env.reset()
        self._agent.learn(
            self.number_of_timesteps,
            callback=callbacks,
            tb_log_name=self.agent.get_next_tensorboard_experiment_name(),
            reset_num_timesteps=False,
        )
        self.save_model()
        self.evaluate_model(validation_environment)

    def export_spline(self, file_name: str) -> None:
        """Exports the current spline to the specified location.

        Args:
            file_name (str): Path to the location where the spline should be stored at.
        """
        self.environment.export_spline(file_name)

    def export_mesh(self, file_name: str) -> None:
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
        if file_name is None:
            path = pathlib.Path(self.save_location)
        elif file_name is not None:
            path = pathlib.Path(file_name)
        else:
            path = pathlib.Path(
                f"default_model_safe_location/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._agent.save(path / "model_end.save")

    def evaluate_model(
        self,
        validation_env: Union[None, Environment] = None,
        throw_error_if_None: bool = False,
    ) -> None:
        """Evaluate the model with the parameters defined in the validation variable. If an agent is already loaded use this agent, else get the agent from the agent variable. Validation will be done inside the 

        Args:
            validation_env (Union[None, Environment], optional): If validation environment already exists it will be used else a new validation evironment will be created. Defaults to None.
            throw_error_if_None (bool, optional): If this is set and the validation variable is None an error is thrown. Defaults to False.

        Raises:
            SbSOvRLValidationNotSet: Thrown if validation is absolutly needed. If not absolutly needed the validation will not be done but no error will be thrown.
        """
        if self.validation:
            if validation_env is None:
                validation_env = self._create_validation_environment(True)
            if self._agent is None:
                self._agent = self.agent.get_agent(validation_env.get_gym_environment())
            reward, episode_length = self.validation.end_validation(
                agent=self._agent, environment=validation_env.get_gym_environment()
            )
            reward_array = np.array(reward)
            mean_reward = reward_array.mean()
            reward_std = reward_array.std()
            log_str = f"The end validation had the following results: mean_reward = {mean_reward}; reward_std = {reward_std}"
            self.get_logger().info(log_str)
            self.get_logger().info(f"The reward per episode was: {reward}.")
            self.get_logger().info(f"The length per episode was: {episode_length}")
        elif throw_error_if_None:
            raise SbSOvRLValidationNotSet()

    def _create_validation_environment(
        self, throw_error_if_None: bool = False
    ) -> Environment:
        """Creates a validation environment.

        Args:
            throw_error_if_None (bool, optional): If this is set and the validation variable is None an error is thrown. Defaults to False.

        Raises:
            SbSOvRLValidationNotSet: Thrown if validation is absolutely needed. If not absolutly needed the validation will not be done but no error will be thrown.

        Returns:
            Environment: Validation environment. Is not a 'gym' environment but and SbSOvRL.parser_environment.Environment. Create the gym environment by calling the function env.get_gym_environment()
        """
        if self.validation is None:
            if (
                throw_error_if_None
            ):  # ok i have no idea why i made this thingy here but i will keep it for posterity
                raise SbSOvRLValidationNotSet()
            return None
        validation_environment = deepcopy(self.environment)

        validation_environment.set_logger_name_recursively(
            self.verbosity._environment_validation_logger
        )
        
        validation_environment.set_validation(
            **self.validation.get_environment_validation_parameters(self.save_location)
        )
        
        return validation_environment
