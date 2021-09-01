from pydantic import BaseModel, confloat, validator
from typing import Literal, Optional, Union, List
import numpy as np
from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.gym_environment import GymEnvironment
from stable_baselines3.ppo import PPO

class Schedule(BaseModel):
    """Exponential Scheduler function is as follows:

    end_value + (start_value-end_value) * e^(exponent * 5 * value) 

    There is an extra clipping step to absolutely ensure that the end value is in between upper and lower bound.

    The constant factor in the exponent (5) is used to better fit the exponential function to the intervall 0-1. For exponential prefactor of -1.
    """
    exponent: confloat(lt=0.) = -1
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
    policy: Literal["MlpPolicy", "MlpPolicy"]
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
    device: str = "auto" # Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[List[List[int]]] = None # additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment) -> PPO:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.

        Returns:
            PPO: Initialized PPO agent.
        """
        return PPO(env = environment, **{k:v for k,v in self.__dict__.items() if k is not 'type'})



AgentTypeDefinition = Union[PPOAgent]