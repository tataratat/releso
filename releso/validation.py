"""File holds the definition class for the validation."""

import pathlib
from typing import Any, Dict, Optional, Tuple

from pydantic.class_validators import validator
from pydantic.types import conint, conlist
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.type_aliases import GymEnv

from releso.base_model import BaseModel
from releso.exceptions import ParserException
from releso.util.logger import get_parser_logger


class Validation(BaseModel):
    """Parser class to define the validation to be performed during training.

    This class is used for the configuration of how validation is to be
    performed during training.
    """

    #: How many timesteps should pass by learning between validation runs
    validation_freq: Optional[conint(ge=1)]
    #: List of validation items. This will be revised later on #TODO
    validation_values: conlist(float, min_items=1)
    #: Whether or not to save the best agent. If agent is not saved only
    #: results will be reported by the agent which produced them will not be
    #: saved.
    save_best_agent: bool
    #: Whether or not to also run a validation when the training is terminated.
    validate_on_training_end: bool
    #: after how many timesteps inside a single episode should the episode be
    #: terminated.
    max_timesteps_in_episode: Optional[conint(ge=1)] = None
    #: Should the episode be terminated if the geometry representation has not
    #: changed between timesteps?
    end_episode_on_geometry_not_changed: bool = False
    #: What reward should be added to the step reward if the geometry was not
    #: changed for the defined number of steps.
    reward_on_geometry_not_changed: Optional[float] = None
    #: What reward should be added to the step reward if the maximal timesteps
    #: per episode is exceeded.
    reward_on_episode_exceeds_max_timesteps: Optional[float] = None

    @validator("reward_on_geometry_not_changed", always=True)
    @classmethod
    def check_if_reward_given_if_geometry_not_change_episode_killer_activated(
        cls, value: float, values: Dict[str, Any]
    ) -> float:
        """Validator for reward_on_geometry_not_changed.

        Checks that 1) if a reward is set, also the boolean value for the
        end_episode_on_geometry_not_changed is True.
        2) If end_episode_on_geometry_not_changed is True a reward value is set

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values
            (here end_episode_on_geometry_not_changed is important)

        Raises:
            ParserException: Error is thrown if one of the conditions is
            not met.

        Returns:
            float: reward for the specified occurrence.
        """
        # due to pydantic this first statement should never be able to happen
        if (
            "end_episode_on_geometry_not_changed" not in values
        ):  # pragma: no cover
            raise ParserException(
                "Validation",
                "reward_on_geometry_not_changed",
                "Could not find definition of parameter "
                "end_episode_on_geometry_not_changed, please define this "
                "variable since otherwise this variable would have no "
                "function.",
            )
        if value is not None and (
            values["end_episode_on_geometry_not_changed"] is None
            or not values["end_episode_on_geometry_not_changed"]
        ):
            raise ParserException(
                "Validation",
                "reward_on_geometry_not_changed",
                "Reward can only be set if end_episode_on_geometry_not_changed"
                " is true.",
            )
        if values["end_episode_on_geometry_not_changed"] and value is None:
            get_parser_logger().warning(
                "Please set a reward value for geometry not changed if episode"
                " should end on it. Will set 0 for you now, but this might not"
                " be you intention."
            )
            value = 0.0
        return value

    @validator("reward_on_episode_exceeds_max_timesteps", always=True)
    @classmethod
    def check_if_reward_given_if_max_steps_killer_activated(
        cls, value: float, values: Dict[str, Any]
    ) -> float:
        """Validator for reward_on_episode_exceeds_max_timesteps.

        Checks that 1) if a reward is set, also that the value for the
        max_timesteps_in_episode is greater than 0.
        2) If max_timesteps_in_episode is greater 0 a reward value is set.

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values
            (here max_timesteps_in_episode is important)

        Raises:
            ParserException: Error is thrown if one of the conditions is
            not met.

        Returns:
            float: reward for the specified occurrence.
        """
        # due to pydantic this first statement should never be able to happen
        if "max_timesteps_in_episode" not in values:  # pragma: no cover
            raise ParserException(
                "Validation",
                "reward_on_episode_exceeds_max_timesteps",
                "Could not find definition of parameter "
                "max_timesteps_in_episode, please define this variable since "
                "otherwise this variable would have no function.",
            )
        if value is not None and (
            values["max_timesteps_in_episode"] is None
            or not values["max_timesteps_in_episode"]
        ):
            raise ParserException(
                "Validation",
                "reward_on_episode_exceeds_max_timesteps",
                "Reward can only be set if max_timesteps_in_episode is a "
                "positive integer.",
            )
        if values["max_timesteps_in_episode"] and value is None:
            get_parser_logger().warning(
                "Please set a reward value for max time steps exceeded, if "
                "episode should end on it. Will set 0 for you now, but this "
                "might not be you intention."
            )
            value = 0.0
        return value

    def should_add_callback(self) -> bool:
        """Bool function whether or not a validation callback is needed.

        Returns:
            bool: Return True if a validation callback is needed else False.
        """
        if self.validation_freq > 0:
            return True
        return False

    def get_callback(
        self,
        eval_environment: GymEnv,
        save_location: Optional[pathlib.Path] = None,
        normalizer_divisor: int = 1,
    ) -> EvalCallback:
        """Creates the EvalCallback with the values given in this object.

        Args:
            eval_environment (GymEnv): Evaluation environment. Should be the
                same as the normal training environment only that here the goal
                values should be set and not random.
            save_location (Optional[pathlib.Path]): Path to where the best
                models should be save to.
            normalizer_divisor (int, optional): Divisor for the eval_freq.
                Defaults to 1.

        Returns:
            EvalCallback: Validation callback parametrized by this object.
        """
        # build the dictionary holding the variables needed to initialize the
        # callback. See callback signature for variable explanation
        variable_dict = {
            "eval_env": eval_environment,
            "n_eval_episodes": len(self.validation_values),
            "eval_freq": int(self.validation_freq / normalizer_divisor),
        }
        # add variables for the case if the best model is to be saved
        if self.save_best_agent and save_location:
            variable_dict["best_model_save_path"] = (
                save_location / "eval/best_model"
            )
            variable_dict["log_path"] = save_location / "eval/log"
        return EvalCallback(**variable_dict)

    def end_validation(
        self, agent: BaseAlgorithm, environment: GymEnv
    ) -> Tuple[float, float]:
        """Function is called at the end of a validation.

        All clean up and last evaluation is going in here.

        Args:
            agent (BaseAlgorithm): Agent which is to be used to validate.
            environment (GymEnv): Validation environment

        Returns:
            Tuple[float, float]: See 'funct'evaluate_policy() for definition
        """
        variable_dict = {
            "model": agent,
            "env": environment,
            "n_eval_episodes": len(self.validation_values),
            "deterministic": True,
            "return_episode_rewards": True,
        }
        return evaluate_policy(**variable_dict)

    def get_environment_validation_parameters(self) -> Dict[str, Any]:
        """Gather the validation arguments used to initialize validator.

        Gets the validation parameters that need to be send to the environment
        if it gets converted to be a validation environment.

        Returns:
            Dict[str, Any]:
                dict with all the necessary parameters. Should mirror the
                parameters in parser_environment.Environment.set_validation
        """
        parameters_to_include = [
            "validation_values",
            "end_episode_on_geometry_not_changed",
            "max_timesteps_in_episode",
            "reward_on_episode_exceeds_max_timesteps",
            "reward_on_geometry_not_changed",
        ]
        return {key: getattr(self, key) for key in parameters_to_include}
