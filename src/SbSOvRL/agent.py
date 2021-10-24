from pathlib import Path
from pydantic import confloat, validator
from typing import Literal, Optional, Union, Dict, Any
# import numpy as np
# from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.gym_environment import GymEnvironment
from pydantic.types import DirectoryPath, FilePath
from stable_baselines3 import PPO, DDPG, SAC, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
import datetime
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.exceptions import SbSOvRLAgentUnknownException

# TODO dynamic agent detection via https://stackoverflow.com/a/3862957
# get argument names https://docs.python.org/3/library/inspect.html
class BaseAgent(SbSOvRL_BaseModel):
    tensorboard_log: Optional[str]

    def get_next_tensorboard_experiment_name(self) -> str:
        if self.tensorboard_log is not None:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return None

class PretrainedAgent(BaseAgent):
    type: Literal["PPO", "SAC", "DDPG"]
    path: FilePath
    tesorboard_run_directory: Union[str, None] = None

    def get_agent(self, environment: GymEnvironment) -> BaseAlgorithm:
        if self.type == "PPO":
            return PPO.load(self.path, environment)
        elif self.type == "DDPG":
            return DDPG.load(self.path, environment)
        elif self.type == "SAC":
            return SAC.load(self.path, environment)
        else:
            raise SbSOvRLAgentUnknownException(self.type)

    def get_next_tensorboard_experiment_name(self) -> str:
        if self.tensorboard_log is not None:
            if self.tesorboard_run_directory:
                return self.tesorboard_run_directory
            else:
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
    optimize_memory_usage: bool = False # Enable a memory efficient variant of the replay buffer at a cost of more complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
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

class DQNAgent(BaseAgent):
    """DQN definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 docu.
    """
    type: Literal["DQN"]
    policy: Literal["MlpPolicy"]
    learning_rate: float = 1e-4
    buffer_size: int = 1000000  # size of the replay buffer
    learning_starts: int = 256 # how many steps of the model to collect transitions for before learning starts
    batch_size: Optional[int] = 32 # Minibatch size for each gradient update
    tau: float = 1.0 # the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    gamma: float = 0.99 # the discount factor
    train_freq: Union[int] = 4 # Update the model every ``train_freq`` steps. 
    gradient_steps: int = 1 #How many gradient steps to do after each rollout (see ``train_freq``) Set to ``-1`` means to do as many gradient steps as steps done in the environment during the rollout.
    optimize_memory_usage: bool = False # Enable a memory efficient variant of the replay buffer at a cost of more complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    target_update_interval: int = 256 # update the target network every ``target_update_interval``
    exploration_fraction: float = 0.1 # fraction of entire training period over which the exploration rate is reduced
    exploration_initial_eps: float = 1.0 # initial value of random action probability
    exploration_final_eps: float = 0.05 # final value of random action probability
    max_grad_norm: float = 10 # The maximum value for the gradient clipping
    seed: Optional[int] = None # Seed for the pseudo random generators
    device: str = "auto" # Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None # additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment) -> DQN:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.

        Returns:
            DQN: Initialized DQN agent.
        """
        return DQN(env = environment, **{k:v for k,v in self.__dict__.items() if not k == 'type'})

AgentTypeDefinition = Union[PPOAgent, DDPGAgent, SACAgent, PretrainedAgent, DQNAgent]