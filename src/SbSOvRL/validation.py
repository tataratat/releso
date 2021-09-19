import pathlib
from SbSOvRL.base_model import SbSOvRL_BaseModel
from pydantic.class_validators import validator
from pydantic.types import conint, conlist
from typing import Optional, List, Tuple
from SbSOvRL.exceptions import SbSOvRLParserException
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.type_aliases import GymEnv
import datetime
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm

class Validation(SbSOvRL_BaseModel):
    validation_freq: Optional[conint(ge=1)]
    validation_values: conlist(float, min_items=1)
    save_best_agent: bool
    validate_on_training_end: bool

    @validator("validation_values")
    @classmethod
    def validate_validation_values_list_not_empty(cls, value: List[float], field: str) -> List[float]:
        if len(value) == 0:
            raise SbSOvRLParserException("Validation", field, "You need to provide validation values")
        return value
    
    def should_add_callback(self) -> bool:
        """Bool function wheter or not a validation callback is needed.

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
            save_locaiton (Optional[pathlib.Path]): Path to where the best models should be save to.
        Returns:
            EvalCallback: Validation callback parametriezed by this object.
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
            "n_eval_episodes": len(self.validation_values)
        }
        return evaluate_policy(**variable_dict)