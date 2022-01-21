"""
File holds the definition class for the validation.
"""
import pathlib
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.util.logger import get_parser_logger
from pydantic.class_validators import validator
from pydantic.fields import Field
from pydantic.types import conint, conlist
from typing import Any, Optional, List, Tuple, Dict
from SbSOvRL.exceptions import SbSOvRLParserException
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm

class Validation(SbSOvRL_BaseModel):
    """
    This class is used for the configuration of how validation is to be performed during training. 
    """
    validation_freq: Optional[conint(ge=1)]     #: How many timesteps should pass by learning between validation runs
    validation_values: conlist(float, min_items=1)  #: List of validation items. This will be revised later on #TODO
    save_best_agent: bool   #: Whether or not to save the best agent. If agent is not saved only results will be reported by the agent which produced them will not be saved.
    validate_on_training_end: bool  #: Whether or not to also run a validation when the training is terminated.
    mesh_base_path_extension: Optional[str] = None  #: Do not know currently
    max_timesteps_in_episode: Optional[conint(ge=1)] = None   #: after how many timesteps inside a single episode should the episode be terminated.
    end_episode_on_spline_not_changed: bool = False  #: Should the episode be terminated if the spline representation has not changed between timesteps?
    reward_on_spline_not_changed: Optional[float] = None    #: What reward should be added to the step reward if the spline was not changed for the defined number of steps.
    reward_on_episode_exceeds_max_timesteps: Optional[float] = None #: What reward should be added to the step reward if the maximal timesteps per episode is exceeded.
    
    @validator("reward_on_spline_not_changed", always=True)
    @classmethod
    def check_if_reward_given_if_spline_not_change_episode_killer_activated(cls, value: float, values: Dict[str, Any]) -> float:
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
            raise SbSOvRLParserException("Validation", "reward_on_spline_not_changed", "Could not find definition of parameter end_episode_on_spline_not_changed, please define this variable since otherwise this variable would have no function.")
        if value is not None and (values["end_episode_on_spline_not_changed"] is None or not values["end_episode_on_spline_not_changed"]):
            raise SbSOvRLParserException("Validation", "reward_on_spline_not_changed", "Reward can only be set if end_episode_on_spline_not_changed is true.")
        if values["end_episode_on_spline_not_changed"] and value is None:
            get_parser_logger().warning("Please set a reward value for spline not changed if episode should end on it. Will set 0 for you now, but this might not be you intention.")
            value = 0.
        return value

    @validator("reward_on_episode_exceeds_max_timesteps", always=True)
    @classmethod
    def check_if_reward_given_if_max_steps_killer_activated(cls, value: float, values: Dict[str, Any]) -> float:
        """Checks that 1) if a reward is set, also that the value for the max_timesteps_in_episode is greater than 0. 2) If max_timesteps_in_episode is greater 0 a reward value is set.

        Args:
            value (float): value to validate
            values (Dict[str, Any]): previously validated values (here max_timesteps_in_episode is important)

        Raises:
            SbSOvRLParserException: Error is thrown if one of the conditions is not met.

        Returns:
            float: reward for the specified occurrence.
        """
        if "max_timesteps_in_episode" not in values:
            raise SbSOvRLParserException("Validation", "reward_on_episode_exceeds_max_timesteps", "Could not find definition of parameter max_timesteps_in_episode, please define this variable since otherwise this variable would have no function.")
        if value is not None and (values["max_timesteps_in_episode"] is None or not values["max_timesteps_in_episode"]):
            raise SbSOvRLParserException("Validation", "reward_on_episode_exceeds_max_timesteps", "Reward can only be set if max_timesteps_in_episode is a positive integer.")
        if values["max_timesteps_in_episode"] and value is None:
            get_parser_logger().warning("Please set a reward value for max time steps exceeded, if episode should end on it. Will set 0 for you now, but this might not be you intention.")
            value = 0.
        return value

    @validator("validation_values")
    @classmethod
    def validate_validation_values_list_not_empty(cls, value: List[float], field: str) -> List[float]:
        """Checks that the validation_values list is not empty.

        Args:
            value (List[float]): value holding the list
            field (str): name of the current field

        Raises:
            SbSOvRLParserException: Thrown if list is empty

        Returns:
            List[float]: validated list
        """
        if len(value) == 0:
            raise SbSOvRLParserException("Validation", field, "You need to provide validation values")
        return value
    
    def should_add_callback(self) -> bool:
        """Bool function whether or not a validation callback is needed.

        Returns:
            bool: Return True if a validation callback is needed else False.
        """
        if self.validation_freq > 0:
            return True
        return False

    def get_callback(self, eval_environment: GymEnv, save_location: Optional[pathlib.Path] = None) -> EvalCallback:
        """Creates the EvalCallback with the values given in this object.

        Args:
            eval_environment (GymEnv): Evaluation environment. Should be the same as the normal training environment only that here the goal values should be set and not random.
            save_location (Optional[pathlib.Path]): Path to where the best models should be save to.
        Returns:
            EvalCallback: Validation callback parametrized by this object.
        """
        variable_dict = {
            "eval_env": eval_environment,
            "n_eval_episodes": len(self.validation_values),
            "eval_freq": self.validation_freq
        }
        if self.save_best_agent and save_location:
            variable_dict["best_model_save_path"] = save_location/"eval/best_model"
            variable_dict["log_path"] = save_location/"eval/log"
        return EvalCallback(**variable_dict)

    def end_validation(self, agent: BaseAlgorithm, environment: GymEnv) -> Tuple[float, float]:
        variable_dict = {
            "model": agent,
            "env": environment,
            "n_eval_episodes": len(self.validation_values),
            "deterministic": True,
            "return_episode_rewards": True
        }
        return evaluate_policy(**variable_dict)

    def get_mesh_base_path(self, base_save_location: pathlib.Path) -> str:
        """Appends the read in base_mesh_path to the save_location path given as input parameter. This makes it so that the validation results are stored inside the results storage of the experiment.

        Args:
            base_save_location (pathlib.Path): Experiment/Trainingsrun save location.

        Returns:
            str: str of path pointing to the location where the meshes from the validation are supposed to be stored
        """
        return str(base_save_location/self.mesh_base_path_extension) if self.mesh_base_path_extension else None

    def get_environment_validation_parameters(self, base_save_location: pathlib.Path) -> Dict[str, Any]:
        """Gets the validation parameters that need to be send to the environment if it gets converted to be a validation environment.

        Args:
            base_save_location (pathlib.Path): Experiment/Trainingsrun save location.

        Returns:
            Dict[str, Any]: dict with all the necessary parameters. Should mirror the parameters in parser_environment.Environment.set_validation
        """
        return {"validation_values": self.validation_values, "base_mesh_path": self.get_mesh_base_path(base_save_location), "end_episode_on_spline_not_change": self.end_episode_on_spline_not_changed, "max_timesteps_in_episode": self.max_timesteps_in_episode, "reward_on_episode_exceeds_max_timesteps": self.reward_on_episode_exceeds_max_timesteps, "reward_on_spline_not_changed": self.reward_on_spline_not_changed}