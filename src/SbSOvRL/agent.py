"""
Out of the box the SbSOvRL package uses agents implemented in the Python package
stable-baselines3. Currently the agents Deep Q-Network (DQN), Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC) and Deep Deterministic Policy Gradient (DDPG)
can be used directly but the others can be added easily. 

The following table shows which agent can be used for which shape optimization approach:

+-------+---------------------------+-----------------------------+
| Agent | Direct shape optimization | Indirect shape optimization |
+=======+===========================+=============================+
| PPO   | YES                       | YES                         |
+-------+---------------------------+-----------------------------+
| DQN   | NO                        | YES                         |
+-------+---------------------------+-----------------------------+
| SAC   | YES                       | NO                          |
+-------+---------------------------+-----------------------------+
| DDPG  | YES                       | NO                          |
+-------+---------------------------+-----------------------------+
| A2C   | Yes                       | YES                         |
+-------+---------------------------+-----------------------------+

Author:
    Clemens Fricke (clemens.fricke@rwth-aachen.de)

"""
from typing import Literal, Optional, Union, Dict, Any
# import numpy as np
# from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.gym_environment import GymEnvironment
from pydantic.types import FilePath
from stable_baselines3 import A2C, PPO, DDPG, SAC, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
import datetime
from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.exceptions import SbSOvRLAgentUnknownException

# TODO dynamic agent detection via https://stackoverflow.com/a/3862957
####################################
# >>> def get_subclasses(type):
# ...     for sub in type.__subclasses__():
# ...             print(sub)
# ...             get_subclasses(sub)
# ... 
# >>> get_subclasses(stable_baselines3.common.base_class.BaseAlgorithm)
# <class 'stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm'>
# <class 'stable_baselines3.a2c.a2c.A2C'>
# <class 'stable_baselines3.ppo.ppo.PPO'>
# <class 'stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm'>
# <class 'stable_baselines3.td3.td3.TD3'>
# <class 'stable_baselines3.ddpg.ddpg.DDPG'>
# <class 'stable_baselines3.dqn.dqn.DQN'>
####################################
# get argument names https://docs.python.org/3/library/inspect.html
class BaseAgent(SbSOvRL_BaseModel):
    """
        The class BaseAgent should be used as the base class for all classes defining agents for the SbSOvRL framework. 
    """
    tensorboard_log: Optional[str] #: base directory of the tensorboard logs if given an experiment name with a current timestamp is also added.

    def get_next_tensorboard_experiment_name(self) -> str:
        """Adds a date and time marker to the tensorboard experiment name so that it can be distinguished from other experiments.

        Returns:
            str: Experiment name consisting of a time and date stamp.
        """
        if self.tensorboard_log is not None:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return None

    def get_additional_kwargs(self, ) -> Dict[str, Any]:
        return {k:v for k,v in self.__dict__.items() if not k in ['type', 'logger_name', 'save_location']}

class PretrainedAgent(BaseAgent):
    """
    This class can be used to load pretrained agents, instead of using untrained agents. Can also be used to only validate this agent without training it further. Please see validation section for this use-case.
    """
    type: Literal["PPO", "SAC", "DDPG", "A2C", "DQN"] #: What RL algorithm was used to train the agent. Needs to be know to correctly load the agent.
    path: FilePath  #: Path to the save files of the pretrained agent.
    tesorboard_run_directory: Union[str, None] = None   #: If the agent is to be trained further the results can be added to the existing tensorboard experiment. This is the path to the existing tenorboard experiment

    def get_agent(self, environment: GymEnvironment, normalizer_divisor: int = 1) -> BaseAlgorithm:
        """Tries to locate the agent defined and to load it correctly.

        Args:
            environment (GymEnvironment): Environment with which the agent will interact.
            normalizer_divisor (int): Currently not used in this function. 

        Raises:
            SbSOvRLAgentUnknownException: Possible exception which can be thrown.

        Returns:
            BaseAlgorithm: Return the correctly loaded agent.
        """
        self.get_logger().info(f"Using pretrained agent of type {self.type} from location {self.path}.")
        if self.type == "PPO":
            return PPO.load(self.path, environment)
        elif self.type == "DDPG":
            return DDPG.load(self.path, environment)
        elif self.type == "SAC":
            return SAC.load(self.path, environment)
        elif self.type == "DQN":
            return DQN.load(self.path, environment)
        elif self.type == "A2C":
            return A2C.load(self.path, environment)
        else:
            raise SbSOvRLAgentUnknownException(self.type)

    def get_next_tensorboard_experiment_name(self) -> str:
        """The tensorboard experiment name of the original training run if given else a new one with the current time stamp.

        Returns:
            str: tensorboard experiment name
        """
        if self.tensorboard_log is not None:
            if self.tesorboard_run_directory:
                return self.tesorboard_run_directory
            else:
                return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return None


class A2CAgent(BaseAgent):
    """PPO definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 documentation.
    """
    type: Literal["A2C"] #: What RL algorithm was used to train the agent. Needs to be know to correctly load the agent.
    policy: Literal["MlpPolicy"]    #: policy defines the network structure which the agent uses
    learning_rate: float = 7e-4 #: The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    n_steps: int = 5 #: The number of steps to run for each environment per update(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization) See https://github.com/pytorch/pytorch/issues/29372
    gamma: float = 0.99 #: Discount factor
    gae_lambda: float = 1.0 #: Factor for trade-off of bias vs variance for Generalized Advantage Estimator to classic advantage when set to 1
    ent_coef: float = 0.0 #: Entropy coefficient for the loss calculation
    vf_coef: float = 0.5 #: Value function coefficient for the loss calculation
    max_grad_norm=0.5 #: The maximum value fot he gradient clipping
    rms_prop_eps=1e-05 #: RMSProp epsilon. It stabilizes square root computation in denominator of of RMSProp update
    use_rms_prop=True #: Whether to use RMSprop (default) or Adam as optimizer
    use_sde=False #: Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
    sde_sample_freq=- 1 #: Sample a new noise matrix every n steps when using gSDE Default: -1 (sample only at begining of roll)
    normalize_advantage=False #: Whether to normalize or not the advantage
    seed: Optional[int] = None #: Seed for the pseudo random generators
    device: str = "auto" #: Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None #: additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment, normalizer_divisor: int = 1) -> A2C:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Notes:
            The A2C variable n_steps scales with the amount of environments the agent is trained with. If this behavior is unwanted you can set the variable ''normalize_training_values'' in BaseParser to true. This will set this variable to the number of environments so that the scaled values can be unscaled.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.
            normalizer_divisor (int): Divides the variable n_steps by this value to descale scaled values with n_environments unequal zero.

        Returns:
            A2C: Initialized A2C agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        self.n_steps = int(self.n_steps/normalizer_divisor)
        return A2C(env = environment, **self.get_additional_kwargs())



class PPOAgent(BaseAgent):
    """PPO definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 documentation.
    """
    type: Literal["PPO"] #: What RL algorithm was used to train the agent. Needs to be know to correctly load the agent.
    policy: Literal["MlpPolicy"]    #: policy defines the network structure which the agent uses
    learning_rate: float = 3e-4 #: The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    n_steps: int = 2048 #: The number of steps to run for each environment per update(i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel) NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization) See https://github.com/pytorch/pytorch/issues/29372
    batch_size: Optional[int] = 64 #: Minibatch size
    n_epochs: int = 10 #: Number of epoch when optimizing the surrogate loss
    gamma: float = 0.99 #: Discount factor
    gae_lambda: float = 0.95 #: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    clip_range: float = 0.2 #: Clipping parameter, it can be a function of the current progress remaining (from 1 to 0).
    ent_coef: float = 0.0 #: Entropy coefficient for the loss calculation
    vf_coef: float = 0.5 #: Value function coefficient for the loss calculation
    seed: Optional[int] = None #: Seed for the pseudo random generators
    device: str = "auto" #: Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None #: additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment, normalizer_divisor: int = 1) -> PPO:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Notes:
            The PPO variable n_steps scales with the amount of environments the agent is trained with. If this behavior is unwanted you can set the variable ''normalize_training_values'' in BaseParser to true. This will set this variable to the number of environments so that the scaled values can be unscaled.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.
            normalizer_divisor (int): Divides the variable n_steps by this value to descale scaled values with n_environments unequal zero.

        Returns:
            PPO: Initialized PPO agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        self.n_steps = int(self.n_steps/normalizer_divisor)
        return PPO(env = environment, **self.get_additional_kwargs())

class DDPGAgent(BaseAgent):
    """DDPG definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 documentation.
    """
    type: Literal["DDPG"] #: What RL algorithm was used to train the agent. Needs to be know to correctly load the agent.
    policy: Literal["MlpPolicy"]    #: policy defines the network structure which the agent uses
    learning_rate: float = 1e-3 #: The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    buffer_size: int = 1000000 #: size of the replay buffer
    learning_starts: int = 100 #: how many steps of the model to collect transitions for before learning starts
    batch_size: Optional[int] = 64 #: Minibatch size
    tau: float = 0.005 #: the soft update coefficient ("Polyak update", between 0 and 1)
    gamma: float = 0.99 #: Discount factor
    optimize_memory_usage: float = 0.95 #: Enable a memory efficient variant of the replay buffer at a cost of more complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#:issuecomment-637501195
    seed: Optional[int] = None #: Seed for the pseudo random generators
    device: str = "auto" #: Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None #: additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment, normalizer_divisor: int = 1) -> DDPG:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.
            normalizer_divisor (int): Currently not used in this function. 

        Returns:
            DDPG: Initialized DDPG agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        return DDPG(env = environment, **self.get_additional_kwargs())

class SACAgent(BaseAgent):
    """SAC definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 documentation.
    """
    type: Literal["SAC"] #: What RL algorithm was used to train the agent. Needs to be know to correctly load the agent.
    policy: Literal["MlpPolicy"]    #: policy defines the network structure which the agent uses
    learning_rate: float = 1e-3 #: The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    buffer_size: int = 1000000  #: size of the replay buffer
    learning_starts: int = 100  #: how many steps of the model to collect transitions for before learning starts
    batch_size: Optional[int] = 64 #: Minibatch size
    tau: float = 0.005 #: the soft update coefficient ("Polyak update", between 0 and 1)
    gamma: float = 0.99 #: Discount factor
    optimize_memory_usage: bool = False #: Enable a memory efficient variant of the replay buffer at a cost of more complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#:issuecomment-637501195
    ent_coef: Union[str, float] = "auto" #: Entropy regularization coefficient. (Equivalent to inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off. Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    target_update_interval: int = 1 #: update the target network every ``target_network_update_freq`` gradient steps.
    target_entropy: Union[str, float] = "auto" #: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    use_sde: bool = False #: Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
    sde_sample_freq: int = -1 #: Sample a new noise matrix every n steps when using gSDE Default: -1 (only sample at the beginning of the rollout)
    use_sde_at_warmup: bool = False #: Whether to use gSDE instead of uniform sampling during the warm up phase (before learning starts)
    seed: Optional[int] = None #: Seed for the pseudo random generators
    device: str = "auto" #: Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None #: additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment, normalizer_divisor: int = 1) -> SAC:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.
            normalizer_divisor (int): Currently not used in this function. 

        Returns:
            SAC: Initialized SAC agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        return SAC(env = environment, **self.get_additional_kwargs())

class DQNAgent(BaseAgent):
    """DQN definition for the stable_baselines3 implementation for this algorithm. Variable comments are taken from the stable_baselines3 documentation.
    """
    type: Literal["DQN"] #: What RL algorithm was used to train the agent. Needs to be know to correctly load the agent.
    policy: Literal["MlpPolicy"]
    learning_rate: float = 1e-4 #: The learning rate, it can be a function of the current progress remaining (from 1 to 0)
    buffer_size: int = 1000000  #: size of the replay buffer
    learning_starts: int = 256 #: how many steps of the model to collect transitions for before learning starts
    batch_size: Optional[int] = 32 #: Minibatch size for each gradient update
    tau: float = 1.0 #: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    gamma: float = 0.99 #: the discount factor
    train_freq: int = 4 #: Update the model every ``train_freq`` steps. 
    gradient_steps: int = 1 #:How many gradient steps to do after each rollout (see ``train_freq``) Set to ``-1`` means to do as many gradient steps as steps done in the environment during the rollout.
    optimize_memory_usage: bool = False #: Enable a memory efficient variant of the replay buffer at a cost of more complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#:issuecomment-637501195
    target_update_interval: int = 256 #: update the target network every ``target_update_interval``
    exploration_fraction: float = 0.1 #: fraction of entire training period over which the exploration rate is reduced
    exploration_initial_eps: float = 1.0 #: initial value of random action probability
    exploration_final_eps: float = 0.05 #: final value of random action probability
    max_grad_norm: float = 10 #: The maximum value for the gradient clipping
    seed: Optional[int] = None #: Seed for the pseudo random generators
    device: str = "auto" #: Device (cpu, cuda, …) on which the code should be run. Setting it to auto, the code will be run on the GPU if possible.
    policy_kwargs: Optional[Dict[str, Any]] = None #: additional arguments to be passed to the policy on creation

    def get_agent(self, environment: GymEnvironment, normalizer_divisor: int = 1) -> DQN:
        """Creates the stable_baselines version of the wanted Agent. Uses all Variables given in the object (except type) as the input parameters of the agent object creation.

        Args:
            environment (GymEnvironment): The environment the agent uses to train.
            normalizer_divisor (int): Currently not used in this function. 

        Returns:
            DQN: Initialized DQN agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        return DQN(env = environment, **self.get_additional_kwargs())

AgentTypeDefinition = Union[PPOAgent, DDPGAgent, SACAgent, PretrainedAgent, DQNAgent, A2CAgent]