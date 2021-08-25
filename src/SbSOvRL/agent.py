from pydantic import BaseModel, confloat, validator
from typing import Literal, Optional, Union, Dict, Any
import numpy as np
from SbSOvRL.exceptions import SbSOvRLParserException

class Schedule(BaseModel):
    """Exponential Scheduler function is as follows:

    end_value + (start_value-end_value) * e^(exponent*value) 

    There is an extra clipping step to absolutely ensure that the end value is in between upper and lower bound.
    """
    exponent: confloat(lt=0.)
    start_value: confloat(gt=0., le=1.)
    end_value: confloat(gt=0., le=1.) = 0

    ###### validators
    @validator("end_value", always=True)
    @classmethod
    def end_value_check_if_lower_than_start_value(cls, v, values, field):
        if "start_value" in values.keys():
            if values["start_value"] > v:
                return v
            else:
                raise SbSOvRLParserException("Schedule", field, "Please ensure that the start_value is higher than the end_value.")
        else:
            raise SbSOvRLParserException("Schedule", field, "Please select a start value.")
    #### object functions

    def __call__(self, value: float) -> float:
        return np.clip(np.exp(self.exponent * value) * (self.start_value - self.end_value) + self.end_value, self.end_value, self.start_value)

class PPOAgent(BaseModel):
    """PPO definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 docu.

    Args:
        BaseModel ([type]): [description]
    """
    type: Literal["PPO"]
    learning_rate: Union[float, Schedule] = 3e-4 # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    n_steps: int = 2048 # The number of steps to run for each environment per update(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization) See https://github.com/pytorch/pytorch/issues/29372
    batch_size: Optional[int] = 64 # Minibatch size
    n_epochs: int = 10 # Number of epoch when optimizing the surrogate loss
    gamma: float = 0.99 # Discount factor
    gae_lambda: float = 0.95 # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    clip_range: Union[float, Schedule] = 0.2 # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
    ent_coef: float = 0.0 # Entropy coefficient for the loss calculation
    vf_coef: float = 0.5 # Value function coefficient for the loss calculation
    seed: Optional[int] = None # Seed for the pseudo random generators



AgentTypeDefinition = Union[PPOAgent]