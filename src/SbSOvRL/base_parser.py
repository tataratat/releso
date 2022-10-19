"""Framework experiment definition and experiment conduction.

File defines the base json object which is needed to define the problem setting
for the command line based usage of the SbSOvRL toolbox/framework.
"""
import datetime
import pathlib
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
from pydantic.fields import Field, PrivateAttr
from pydantic.types import conint
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from SbSOvRL.agent import AgentTypeDefinition
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.callback import EpisodeLogCallback
from SbSOvRL.exceptions import SbSOvRLValidationNotSet
from SbSOvRL.parser_environment import Environment
from SbSOvRL.validation import Validation
from SbSOvRL.verbosity import Verbosity


class BaseParser(SbSOvRL_BaseModel):
    """Class parses the experiment definition and conducts the training.

    This class can be used to initialize the SbSOvRL Framework from the command
    line by reading in a json representation of the Spline based Shape
    Optimization problem which is to be solved via Reinforcement Learning.
    """
    #: Defining the verbosity of the training process and environment loading
    verbosity: Verbosity = Field(default_factory=Verbosity)
    #: Definition of the agent to be used during training and/or validation of
    #: the RL use case
    agent: AgentTypeDefinition
    #: Definition of the environment which encodes the parameters of the RL use
    #: case
    environment: Environment
    #: Number of timesteps the training process should run for
    number_of_timesteps: conint(
        ge=1
    )
    #: Number of episodes the training process should run for. If given both
    #: timesteps and max episodes can stop the trainings progress.
    #: Default: None
    number_of_episodes: Optional[conint(
        ge=1
    )] = None
    #: Definition of the validation parameters
    validation: Optional[Validation]
    #: Number of environments to train in parallel
    n_environments: Optional[conint(ge=1)]
    # : Should training parameters be normalized to the number of environments
    normalize_training_values: bool = False
    #: Should the multi environment be run sequentially (True) or with multi
    #: processing (False)
    multi_env_sequential: bool = False

    # internal objects
    #: Holds the trainable agent for the RL use case. The
    #: SbSOvRL.base_parser.BaseParser.agent defines the type and parameters of
    #: the agent this is the actual trainable agent.
    _agent: Optional[BaseAlgorithm] = PrivateAttr(
        default=None
    )

    def __init__(__pydantic_self__, **data: Any) -> None:
        """Constructor of the base parser.

        Initializes the class correctly and also adds the correct logger to all
        subclasses.
        """
        super().__init__(**data)
        __pydantic_self__.set_logger_name_recursively(
            __pydantic_self__.verbosity._environment_logger
        )

    def learn(self) -> None:
        """Starts the training that is specified in the loaded json file."""
        train_env: Optional[VecEnv] = None
        validation_environment = self._create_validation_environment()
        normalizer_divisor = 1 if self.n_environments is None or not \
            self.normalize_training_values else self.n_environments
        if self.n_environments and self.n_environments > 1:
            env_create_list = []
            for idx in range(self.n_environments):
                env_create_list.append(self._create_new_environment(
                    self.get_logger().name+f"_{idx}"))
            if self.multi_env_sequential:
                train_env = DummyVecEnv(env_create_list)
            else:
                train_env = SubprocVecEnv(env_create_list)
        else:
            train_env = DummyVecEnv(
                [lambda: self.environment.get_gym_environment()])

        self._agent = self.agent.get_agent(train_env, normalizer_divisor)
        self.get_logger().info(f"Agent is of type {type(self._agent)}")
        callbacks = [
            EpisodeLogCallback(
                episode_log_location=self.save_location / "episode_log.csv",
                verbose=1
            ),
        ]
        if self.number_of_episodes is not None:
            num = self.number_of_episodes
            if self.normalize_training_values:
                num = int(num/normalizer_divisor)
            callbacks.append(StopTrainingOnMaxEpisodes(max_episodes=num))

        if self.validation:
            if self.validation.should_add_callback():
                callbacks.append(
                    self.validation.get_callback(
                        validation_environment.get_gym_environment(),
                        save_location=self.save_location,
                        normalizer_divisor=normalizer_divisor
                    )
                )

        self.get_logger().info(
            f"The environment is now trained for {self.number_of_episodes} "
            f"episodes or {self.number_of_timesteps} timesteps."
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
            file_name (str): Path to the location where the spline should be
            stored at.
        """
        self.environment.export_spline(file_name)

    def export_mesh(self, file_name: str) -> None:
        """Exports the current mesh to the specified location.

        Args:
            file_name (str): Path to the location where the mesh should be
            stored at.
        """
        self.environment.export_mesh(file_name)

    def save_model(self, file_name: Optional[str] = None) -> None:
        """Save the state of the agent.

        Saves the current agent to the specified location or to a default
        location.

        Args:
            file_name (Optional[str]): Path where the agent is saved to. If
            None will see if json had a save location if also not given will
            use a default location. Defaults to None.
        """
        if file_name is None:
            path = pathlib.Path(self.save_location)
        elif file_name is not None:
            path = pathlib.Path(file_name)
        else:
            path = pathlib.Path(
                f"default_model_safe_location/"
                f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._agent.save(path / "model_end.save")

    def evaluate_model(
        self,
        validation_env: Union[None, Environment] = None,
        throw_error_if_None: bool = False,
    ) -> None:
        """Validate the current agent.

        Evaluate the model with the parameters defined in the validation
        variable. If an agent is already loaded use this agent, else get the
        agent from the agent variable. Validation will be done inside the

        Args:
            validation_env (Union[None, Environment], optional): If validation
            environment already exists it will be used else a new validation
            environment will be created. Defaults to None.
            throw_error_if_None (bool, optional): If this is set and the
            validation variable is None an error is thrown. Defaults to False.

        Raises:
            SbSOvRLValidationNotSet: Thrown if validation is absolutely needed.
            If not absolutely needed the validation will not be done but no
            error will be thrown.
        """
        if self.validation:
            if validation_env is None:
                validation_env = self._create_validation_environment(True)
            if self._agent is None:
                self._agent = self.agent.get_agent(
                    validation_env.get_gym_environment())
            reward, episode_length = self.validation.end_validation(
                agent=self._agent,
                environment=validation_env.get_gym_environment()
            )
            reward_array = np.array(reward)
            mean_reward = reward_array.mean()
            reward_std = reward_array.std()
            log_str = f"The end validation had the following results: "
            f"mean_reward = {mean_reward}; reward_std = {reward_std}"
            self.get_logger().info(log_str)
            self.get_logger().info(f"The reward per episode was: {reward}.")
            self.get_logger().info(
                f"The length per episode was: {episode_length}")
        elif throw_error_if_None:
            raise SbSOvRLValidationNotSet()

    def _create_new_environment(self, logger_name: str):
        """Function used for multi environment training.

        Args:
            logger_name (str): name of the logger used for the rl_logger used
            for this specific environment.
        """
        def _init():
            c_env = deepcopy(self.environment)
            logging_information = {
                "logger_name": logger_name,
                "log_file_location": self.verbosity.SbSOvRL_logfile_location,
                "logging_level": self.verbosity.environment
            }
            return c_env.get_gym_environment(
                logging_information=logging_information
            )
        return _init

    def _create_validation_environment(
        self, throw_error_if_None: bool = False
    ) -> Environment:
        """Creates a validation environment.

        Args:
            throw_error_if_None (bool, optional):
                If this is set and the validation variable is None an error is
                thrown. Defaults to False.

        Raises:
            SbSOvRLValidationNotSet:
                Thrown if validation is absolutely needed. If not absolutely
                needed the validation will not be done but no error will be
                thrown.

        Returns:
            Environment:
                Validation environment. Is not a 'gym' environment but and
                SbSOvRL.parser_environment.Environment. Create the gym
                environment by calling the function env.get_gym_environment()
        """
        if self.validation is None:
            # ok i have no idea why i made this thingy here but i will keep it
            # for posterity
            if throw_error_if_None:
                raise SbSOvRLValidationNotSet()
            return None
        validation_environment = deepcopy(self.environment)

        validation_environment.set_logger_name_recursively(
            self.verbosity._environment_validation_logger
        )
        validation_environment._id = None
        validation_environment.set_validation(
            **self.validation.get_environment_validation_parameters(
                self.save_location)
        )

        return validation_environment
