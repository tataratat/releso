"""Framework experiment definition and experiment conduction.

File defines the base json object which is needed to define the problem setting
for the command line based usage of the ReLeSO toolbox/framework.
"""

import pathlib
from copy import deepcopy
from typing import Any, Optional, Union

import numpy as np
from pydantic.fields import Field, PrivateAttr
from pydantic.types import conint
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from releso.agent import AgentTypeDefinition
from releso.base_model import BaseModel
from releso.callback import EpisodeLogCallback, StepLogCallback
from releso.exceptions import ValidationNotSet
from releso.parser_environment import Environment
from releso.validation import Validation
from releso.verbosity import Verbosity


class BaseParser(BaseModel):
    """Class parses the experiment definition and conducts the training.

    This class can be used to initialize the ReLeSO Framework from the command
    line by reading in a json representation of the RL based Shape
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
    number_of_timesteps: conint(ge=1)
    #: Number of episodes the training process should run for. If given both
    #: timesteps and max episodes can stop the trainings progress.
    #: Default: None
    number_of_episodes: Optional[conint(ge=1)] = None
    #: Definition of the validation . Defaults to None.
    validation: Optional[Validation] = None
    #: Number of environments to train in parallel. Defaults to None.
    n_environments: Optional[conint(ge=1)] = 1
    #: Should training parameters be normalized to the number of environments?
    #: If True the number of steps between learnings are divided by the number
    #: of environments. This increases the training speed for PPO and A2C

    #: but the training might be a little bit more unstable.
    #: Defaults to False.
    normalize_training_values: bool = False
    #: Should the multi environment be run sequentially (True) or with multi
    #: processing (False). Defaults to False.
    multi_env_sequential: bool = False
    #: Number of episodes after which the episode_log is updated. It will be
    #: updated at the end of the training in any case. But making this number
    #: higher will lower the computational overhead. Defaults to 100.
    episode_log_update: conint(ge=1) = 100
    #: Flag indicating whether the step information (like actions,
    #: observations, ...) should be logged to file. Defaults to False.
    export_step_log: bool = False
    #: Number of steps after which the step_log is updated. It will be
    #: updated at the end of the training in any case. But making this number
    #: higher will lower the computational overhead. Defaults to 0 which
    #: triggers the output after every episode.
    step_log_update: conint(ge=0) = 0
    #: Flag indicating whether the step_log should also contain the
    #: information of the environment step. Defaults to False.
    step_log_info: bool = False

    # internal objects
    #: Holds the trainable agent for the RL use case. The
    #: ReLeSO.base_parser.BaseParser.agent defines the type and parameters of
    #: the agent this is the actual trainable agent.
    _agent: Optional[BaseAlgorithm] = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        """Constructor of the base parser.

        Initializes the class correctly and also adds the correct logger to all
        subclasses.
        """
        super().__init__(**data)
        self.set_logger_name_recursively(self.verbosity._environment_logger)

    def learn(self) -> None:
        """Starts the training that is specified in the loaded json file.

        Successive calls to this function does not train the agent further

        but reinitialize the agent.
        """
        train_env: Optional[VecEnv] = None
        validation_environment = self._create_validation_environment()
        normalizer_divisor = (
            1
            if self.n_environments is None
            or not self.normalize_training_values
            else self.n_environments
        )
        if self.n_environments and self.n_environments > 1:
            env_create_list = [
                self._create_new_environment(
                    self.get_logger().name + f"_{idx}"
                )
                for idx in range(self.n_environments)
            ]
            if self.multi_env_sequential:
                train_env = DummyVecEnv(env_create_list)
            else:
                train_env = SubprocVecEnv(env_create_list)
        else:
            train_env = DummyVecEnv([
                lambda: self.environment.get_gym_environment()
            ])

        self._agent = self.agent.get_agent(train_env, normalizer_divisor)
        self.get_logger().info(f"Agent is of type {type(self._agent)}")
        callbacks = [
            EpisodeLogCallback(
                episode_log_location=self.save_location / "episode_log.csv",
                verbose=1,
                update_n_episodes=self.episode_log_update,
                logger=self.get_logger(),
            ),
        ]

        if self.export_step_log:
            callbacks.append(
                StepLogCallback(
                    step_log_location=self.save_location / "step_log.jsonl",
                    verbose=1,
                    update_n_steps=self.step_log_update,
                    log_infos=self.step_log_info,
                    logger=self.get_logger(),
                ),
            )

        if self.number_of_episodes is not None:
            num = self.number_of_episodes
            if self.normalize_training_values:
                num = int(num / normalizer_divisor)
            callbacks.append(StopTrainingOnMaxEpisodes(max_episodes=num))

        if self.validation:
            if self.validation.should_add_callback():
                callbacks.append(
                    self.validation.get_callback(
                        validation_environment.get_gym_environment(),
                        save_location=self.save_location,
                        normalizer_divisor=normalizer_divisor,
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

    def save_model(self, file_name: Optional[str] = None) -> str:
        """Save the state of the agent.

        Saves the current agent to the specified location or to a default
        location.

        Args:
            file_name (Optional[str]): Path where the agent is saved to. If
            None will see if json had a save location if also not given will
            use a default location. Defaults to None.

        Returns:
            str: Path where the agent was saved to.
        """
        if self._agent is None:
            raise RuntimeError("Please train the agent first.")
        if file_name is None:
            path = pathlib.Path(self.save_location)
        else:
            path = pathlib.Path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_path = path / "model_end.save"
        self._agent.save(save_path)
        return str(save_path)

    def evaluate_model(
        self,
        validation_env: Union[None, Environment] = None,
        throw_error_if_none: bool = False,
    ) -> None:
        """Validate the current agent.

        Evaluate the model with the parameters defined in the validation
        variable. If an agent is already loaded use this agent, else get the
        agent from the agent variable. Validation will be done inside the

        Args:
            validation_env (Union[None, Environment], optional): If validation
            environment already exists it will be used else a new validation
            environment will be created. Defaults to None.
            throw_error_if_none (bool, optional): If this is set and the
            validation variable is None an error is thrown. Defaults to False.

        Raises:
            ValidationNotSet: Thrown if validation is absolutely needed.
            If not absolutely needed the validation will not be done but no
            error will be thrown.
        """
        if self.validation:
            if validation_env is None:
                validation_env = self._create_validation_environment(True)
            if self._agent is None:
                self._agent = self.agent.get_agent(
                    validation_env.get_gym_environment()
                )
            reward, episode_length = self.validation.end_validation(
                agent=self._agent,
                environment=validation_env.get_gym_environment(),
            )
            reward_array = np.array(reward)
            mean_reward = reward_array.mean()
            reward_std = reward_array.std()
            log_str = (
                f"The end validation had the following results: "
                f"mean_reward = {mean_reward}; reward_std = {reward_std}"
            )
            self.get_logger().info(log_str)
            self.get_logger().info(f"The reward per episode was: {reward}.")
            self.get_logger().info(
                f"The length per episode was: {episode_length}"
            )
        elif throw_error_if_none:
            raise ValidationNotSet

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
                "log_file_location": self.verbosity.logfile_location,
                "logging_level": self.verbosity.environment,
            }
            return c_env.get_gym_environment(
                logging_information=logging_information
            )

        return _init

    def _create_validation_environment(
        self, throw_error_if_none: bool = False
    ) -> Optional[Environment]:
        """Creates a validation environment.

        Args:
            throw_error_if_none (bool, optional):
                If this is set and the validation variable is None an error is
                thrown. Defaults to False.

        Raises:
            ValidationNotSet:
                Thrown if validation is absolutely needed. If not absolutely
                needed the validation will not be done but no error will be
                thrown.

        Returns:
            Environment:
                Validation environment. Is not a 'gym' environment but and
                ReLeSO.parser_environment.Environment. Create the gym
                environment by calling the function env.get_gym_environment()
        """
        if self.validation is None:
            # ok i have no idea why i made this thingy here but i will keep it
            # for posterity
            if throw_error_if_none:
                raise ValidationNotSet
            return None
        validation_environment = deepcopy(self.environment)

        validation_environment.set_logger_name_recursively(
            self.verbosity._environment_validation_logger
        )
        validation_environment._id = None
        validation_environment.set_validation(
            **self.validation.get_environment_validation_parameters()
        )

        return validation_environment
