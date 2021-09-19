from pathlib import Path
from pydantic import confloat, validator
from typing import Literal, Optional, Union, Dict, Any
# import numpy as np
# from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.gym_environment import GymEnvironment
from stable_baselines3 import PPO, DDPG, SAC
import datetime
from SbSOvRL.base_model import SbSOvRL_BaseModel

# class Schedule(SbSOvRL_BaseModel):
#     """Exponential Scheduler function is as follows:

#     end_value + (start_value-end_value) * e^(exponent * 5 * value) 

#     There is an extra clipping step to absolutely ensure that the end value is in between upper and lower bound.

#     The constant factor in the exponent (5) is used to better fit the exponential function to the interval 0-1. For exponential prefactor of -1.
#     """
#     exponent: confloat(lt=0.) = -1
#     start_value: confloat(gt=0., le=1.)
#     end_value: confloat(gt=0., le=1.) = 0

#     ###### validators
#     @validator("end_value", always=True)
#     @classmethod
#     def end_value_check_if_lower_than_start_value(cls, v, values, field):
#         if "start_value" in values.keys():
#             if values["start_value"] > v:
#                 return v
#             else:
#                 raise SbSOvRLParserException("Schedule", field, "Please ensure that the start_value is higher than the end_value.")
#         else:
#             raise SbSOvRLParserException("Schedule", field, "Please select a start value.")
#     #### object functions

#     def __call__(self, value: float) -> float:
#         return np.clip(np.exp(self.exponent * value) * (self.start_value - self.end_value) + self.end_value, self.end_value, self.start_value)

class BaseAgent(SbSOvRL_BaseModel):
    tensorboard_log: Optional[str]

    def get_next_tensorboard_experiment_name(self) -> str:
        if self.tensorboard_log is not None:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return None




class PPOAgent(BaseAgent):
    """PPO definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 docu.
    """
    type: Literal["PPO"]
    policy: Literal["MlpPolicy"]
    learning_rate: Union[float] = 3e-4 # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    n_steps: int = 2048 # The number of steps to run for each environment per update(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization) See https://github.com/pytorch/pytorch/issues/29372
    batch_size: Optional[int] = 64 # Minibatch size
    n_epochs: int = 10 # Number of epoch when optimizing the surrogate loss
    gamma: float = 0.99 # Discount factor
    gae_lambda: float = 0.95 # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    clip_range: Union[float] = 0.2 # Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
    ent_coef: float = 0.0 # Entropy coefficient for the loss calculation
    vf_coef: float = 0.5 # Value function coefficient for the loss calculation
    seed: Optional[int] = None # Seed for the pseudo random generators
    device: str = "auto" # Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None # additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment) -> PPO:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.

        Returns:
            PPO: Initialized PPO agent.
        """
        return PPO(env = environment, **{k:v for k,v in self.__dict__.items() if not k == 'type'})

class DDPGAgent(BaseAgent):
    """DDPG definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 docu.
    """
    type: Literal["DDPG"]
    policy: Literal["MlpPolicy"]
    learning_rate: Union[float] = 1e-3 # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    buffer_size: int = 1000000 # size of the replay buffer
    learning_starts: int = 100 # how many steps of the model to collect transitions for before learning starts
    batch_size: Optional[int] = 64 # Minibatch size
    tau: float = 0.005 # the soft update coefficient ("Polyak update", between 0 and 1)
    gamma: float = 0.99 # Discount factor
    optimize_memory_usage: float = 0.95 # Enable a memory efficient variant of the replay buffer at a cost of more complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    seed: Optional[int] = None # Seed for the pseudo random generators
    device: str = "auto" # Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None # additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment) -> DDPG:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.

        Returns:
            DDPG: Initialized DDPG agent.
        """
        return DDPG(env = environment, **{k:v for k,v in self.__dict__.items() if not k == 'type'})

class SACAgent(BaseAgent):
    """SAC definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 docu.
    """
    type: Literal["SAC"]
    policy: Literal["MlpPolicy"]
    learning_rate: Union[float] = 1e-3 # The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    buffer_size: int = 1000000 # size of the replay buffer
    learning_starts: int = 100 # how many steps of the model to collect transitions for before learning starts
    batch_size: Optional[int] = 64 # Minibatch size
    tau: float = 0.005 # the soft update coefficient ("Polyak update", between 0 and 1)
    gamma: float = 0.99 # Discount factor
    optimize_memory_usage: float = 0.95 # Enable a memory efficient variant of the replay buffer at a cost of more complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    ent_coef: Union[str, float] = "auto" # Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off. Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    target_update_interval: int = 1 # update the target network every ``target_network_update_freq`` gradient steps.
    target_entropy: Union[str, float] = "auto" # target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    use_sde: bool = False # Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
    sde_sample_freq: int = -1 # Sample a new noise matrix every n steps when using gSDE Default: -1 (only sample at the beginning of the rollout)
    use_sde_at_warmup: bool = False # Whether to use gSDE instead of uniform sampling during the warm up phase (before learning starts)
    seed: Optional[int] = None # Seed for the pseudo random generators
    device: str = "auto" # Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None # additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment) -> SAC:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.

        Returns:
            SAC: Initialized SAC agent.
        """
        return SAC(env = environment, **{k:v for k,v in self.__dict__.items() if not k == 'type'})



AgentTypeDefinition = Union[PPOAgent, DDPGAgent, SACAgent]