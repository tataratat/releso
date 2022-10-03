"""
Out of the box the SbSOvRL package uses agents implemented in the Python
package stable-baselines3. Currently the agents Deep Q-Network (DQN), Proximal
Policy Optimization (PPO), Soft Actor-Critic (SAC) Advantage Actor Critic (A2C)
and Deep Deterministic Policy Gradient (DDPG) can be used directly but the
others can be added easily.

The following table shows which agent can be used for which shape optimization
approach:

+-------+---------------------------+--------------------------------+
| Agent | Direct shape optimization | Incremental shape optimization |
+=======+===========================+================================+
| PPO   | YES                       | YES                            |
+-------+---------------------------+--------------------------------+
| DQN   | NO                        | YES                            |
+-------+---------------------------+--------------------------------+
| SAC   | YES                       | NO                             |
+-------+---------------------------+--------------------------------+
| DDPG  | YES                       | NO                             |
+-------+---------------------------+--------------------------------+
| A2C   | Yes                       | YES                            |
+-------+---------------------------+--------------------------------+

Author:
    Clemens Fricke (clemens.david.fricke@tuwien.ac.at)

"""
import datetime
from typing import Any, Dict, Literal, Optional, Union

from pydantic.types import FilePath
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from SbSOvRL.base_model import SbSOvRL_BaseModel
from SbSOvRL.exceptions import SbSOvRLAgentUnknownException
from SbSOvRL.feature_extractor import (SbSOvRL_CombinedExtractor,
                                       SbSOvRL_FeatureExtractor)
# import numpy as np
# from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.gym_environment import GymEnvironment

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
        The class BaseAgent should be used as the base class for all classes
        defining agents for the SbSOvRL framework.
    """
    #: base directory of the tensorboard logs if given an experiment name
    #: with a current timestamp is also added.
    tensorboard_log: Optional[str]

    def get_next_tensorboard_experiment_name(self) -> str:
        """
        Adds a date and time marker to the tensorboard experiment name so that
        it can be distinguished from other experiments.

        Returns:
            str: Experiment name consisting of a time and date stamp.
        """
        if self.tensorboard_log is not None:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return None


class BaseTrainingAgent(BaseAgent):
    """
        The class BaseAgent should be used as the base class for all classes
        defining agents for the SbSOvRL framework.
    """
    #: policy defines the network structure which the agent uses
    policy: Literal["MlpPolicy", "CnnPolicy", "MultiInputPolicy"]
    #: If given the str identifies the Custom Feature Extractor to be added.
    use_custom_feature_extractor: Optional[Literal["resnet18", "mobilenetv2",
                                                   "mobilenetv3_small",
                                                   "mobilenetv3_large"]] = None
    #: use the custom feature extractor with out a final linear layer
    cfe_without_linear: bool = False
    #: additional arguments to be passed to the policy on creation
    policy_kwargs: Optional[Dict[str, Any]] = None

    def get_next_tensorboard_experiment_name(self) -> str:
        """
        Adds a date and time marker to the tensorboard experiment name so that
        it can be distinguished from other experiments.

        Returns:
            str: Experiment name consisting of a time and date stamp.
        """
        if self.tensorboard_log is not None:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return None

    def get_additional_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Reads and gets the additional keyword arguments for the agent
        definition.

        Returns:
            Dict[str, Any]: Dictionary of the needed additional keywords.
        """
        if self.policy_kwargs is None:
            self.policy_kwargs = {}
        if self.use_custom_feature_extractor:
            self.policy_kwargs["features_extractor_kwargs"] = dict()
            if self.use_custom_feature_extractor:
                if self.policy == "CnnPolicy":
                    self.policy_kwargs["features_extractor_class"] = \
                        SbSOvRL_FeatureExtractor
                elif self.policy == "MultiInputPolicy":
                    self.policy_kwargs["features_extractor_class"] = \
                        SbSOvRL_CombinedExtractor
                else:
                    self.get_logger().warning(
                        "Please use the CnnPolicy or the MultiInputPolicy with"
                        " the SbSOvRL_FeatureExtractors everything else might "
                        "not work. But I also have no idea what works and what"
                        " not. Will try to use the "
                        "SbSOvRL_CustomFeatureExtractor, but as said might not"
                        " work with the current policy.")
                    self.policy_kwargs["features_extractor_class"] = \
                        SbSOvRL_FeatureExtractor
                if self.cfe_without_linear:
                    self.policy_kwargs["features_extractor_kwargs"][
                        "without_linear"] = True
                self.policy_kwargs["features_extractor_kwargs"][
                    "network_type"] = self.use_custom_feature_extractor
                self.policy_kwargs["features_extractor_kwargs"]["logger"] = \
                    self.get_logger()
        return_dict = {
            k: v
            for k, v in self.__dict__.items() if k not in [
                'type', 'logger_name', 'save_location',
                'use_custom_feature_extractor', "cfe_without_linear"
            ]
        }
        return return_dict


class PretrainedAgent(BaseAgent):
    """
    This class can be used to load pretrained agents, instead of using
    untrained agents. Can also be used to only validate this agent without
    training it further. Please see validation section for this use-case.
    """
    #: What RL algorithm was used to train the agent. Needs to be know to
    #: correctly load the agent.
    type: Literal["PPO", "SAC", "DDPG", "A2C", "DQN"]
    #: Path to the save files of the pretrained agent.
    path: FilePath
    #: If the agent is to be trained further the results can be added to the
    # existing tensorboard experiment. This is the path to the existing
    # tensorboard experiment
    tesorboard_run_directory: Union[str, None] = None

    def get_agent(self,
                  environment: GymEnvironment,
                  normalizer_divisor: int = 1) -> BaseAlgorithm:
        """Tries to locate the agent defined and to load it correctly.

        Args:
            environment (GymEnvironment):
                Environment with which the agent will interact.
            normalizer_divisor (int): Currently not used in this function.

        Raises:
            SbSOvRLAgentUnknownException:
                Possible exception which can be thrown.

        Returns:
            BaseAlgorithm: Return the correctly loaded agent.
        """
        self.get_logger().info(
            f"Using pretrained agent of type {self.type} from location "
            f"{self.path}.")
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
        """
        The tensorboard experiment name of the original training run if given
        else a new one with the current time stamp.

        Returns:
            str: tensorboard experiment name
        """
        if self.tensorboard_log is not None:
            if self.tesorboard_run_directory:
                return self.tesorboard_run_directory
            else:
                return super().get_next_tensorboard_experiment_name()
        return None


class A2CAgent(BaseTrainingAgent):
    """
    PPO definition for the stable_baselines3 implementation for this algorithm.
    Variable comments are taken from the stable_baselines3 documentation.
    """
    #: What RL algorithm was used to train the agent. Needs to be know to
    #: correctly load the agent.
    type: Literal["A2C"]
    #: The learning rate, it can be a function of the current progress
    #: remaining (from 1 to 0)
    learning_rate: float = 7e-4
    #: The number of steps to run for each environment per update(i.e. rollout
    #: buffer size is n_steps * n_envs where n_envs is number of environment
    #: copies running in parallel) NOTE: n_steps * n_envs must be greater than
    #: 1 (because of the advantage normalization)
    #: See https://github.com/pytorch/pytorch/issues/29372
    n_steps: int = 5
    #: Discount factor
    gamma: float = 0.99
    #: Factor for trade-off of bias vs variance for Generalized Advantage
    #: Estimator to classic advantage when set to 1
    gae_lambda: float = 1.0
    #: Entropy coefficient for the loss calculation
    ent_coef: float = 0.0
    #: Value function coefficient for the loss calculation
    vf_coef: float = 0.5
    #: The maximum value fot he gradient clipping
    max_grad_norm = 0.5
    #: RMSProp epsilon. It stabilizes square root computation in denominator
    #: of of RMSProp update
    rms_prop_eps = 1e-05
    #: Whether to use RMSprop (default) or Adam as optimizer
    use_rms_prop = True
    #: Whether to use generalized State Dependent Exploration (gSDE) instead
    #: of action noise exploration (default: False)
    use_sde = False
    #: Sample a new noise matrix every n steps when using gSDE Default: -1
    #: (sample only at beginning of roll)
    sde_sample_freq = -1
    #: Whether to normalize or not the advantage
    normalize_advantage = False
    #: Seed for the pseudo random generators
    seed: Optional[int] = None
    #: Device (cpu, cuda, …) on which the code should be run. Setting it to
    #: auto, the code will be run on the GPU if possible.
    device: str = "auto"

    def get_agent(self,
                  environment: GymEnvironment,
                  normalizer_divisor: int = 1) -> A2C:
        """
        Creates the stable_baselines version of the wanted Agent. Uses all
        Variables given in the object (except type) as the input parameters of
        the agent object creation.

        Notes:
            The A2C variable n_steps scales with the amount of environments the
            agent is trained with. If this behavior is unwanted you can set the
            variable ''normalize_training_values'' in BaseParser to true. This
            will set this variable to the number of environments so that the
            scaled values can be unscaled.

        Args:
            environment (GymEnvironment):
                The environment the agent uses to train.
            normalizer_divisor (int):
                Divides the variable n_steps by this value to descale scaled
                values with n_environments unequal zero.

        Returns:
            A2C: Initialized A2C agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        self.n_steps = int(self.n_steps / normalizer_divisor)
        return A2C(env=environment, **self.get_additional_kwargs())


class PPOAgent(BaseTrainingAgent):
    """
    PPO definition for the stable_baselines3 implementation for this algorithm.
    Variable comments are taken from the stable_baselines3 documentation.
    """
    #: What RL algorithm was used to train the agent. Needs to be know to
    #: correctly load the agent.
    type: Literal["PPO"]
    #: The learning rate, it can be a function of the current progress
    #: remaining (from 1 to 0)
    learning_rate: float = 3e-4
    #: The number of steps to run for each environment per update(i.e.
    #: rollout buffer size is n_steps * n_envs where n_envs is number of
    #: environment copies running in parallel) NOTE: n_steps * n_envs must be
    #: greater than 1 (because of the advantage normalization) See
    #: https://github.com/pytorch/pytorch/issues/29372
    n_steps: int = 2048
    #: Minibatch size
    batch_size: Optional[int] = 64
    #: Number of epoch when optimizing the surrogate loss
    n_epochs: int = 10
    #: Discount factor
    gamma: float = 0.99
    #: Factor for trade-off of bias vs variance for Generalized Advantage
    #: Estimator
    gae_lambda: float = 0.95
    #: Clipping parameter, it can be a function of the current progress
    #: remaining (from 1 to 0).
    clip_range: float = 0.2
    #: Entropy coefficient for the loss calculation
    ent_coef: float = 0.0
    #: Value function coefficient for the loss calculation
    vf_coef: float = 0.5
    #: Seed for the pseudo random generators
    seed: Optional[int] = None
    #: Device (cpu, cuda, …) on which the code should be run. Setting it to
    #: auto, the code will be run on the GPU if possible.
    device: str = "auto"

    def get_agent(self,
                  environment: GymEnvironment,
                  normalizer_divisor: int = 1) -> PPO:
        """
        Creates the stable_baselines version of the wanted Agent. Uses all
        Variables given in the object (except type) as the input parameters of
        the agent object creation.

        Notes:
            The PPO variable n_steps scales with the amount of environments the
            agent is trained with. If this behavior is unwanted you can set the
            variable ''normalize_training_values'' in BaseParser to true. This
            will set this variable to the number of environments so that the
            scaled values can be unscaled.

        Args:
            environment (GymEnvironment):
                The environment the agent uses to train.
            normalizer_divisor (int):
                Divides the variable n_steps by this value to descale scaled
                values with n_environments unequal zero.

        Returns:
            PPO: Initialized PPO agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        self.n_steps = int(self.n_steps / normalizer_divisor)
        return PPO(env=environment, **self.get_additional_kwargs())


class DDPGAgent(BaseTrainingAgent):
    """
    DDPG definition for the stable_baselines3 implementation for this
    algorithm. Variable comments are taken from the stable_baselines3
    documentation.
    """
    #: What RL algorithm was used to train the agent. Needs to be know to
    #: correctly load the agent.
    type: Literal["DDPG"]
    #: The learning rate, it can be a function of the current progress
    #: remaining (from 1 to 0)
    learning_rate: float = 1e-3
    #: size of the replay buffer
    buffer_size: int = 100000
    #: how many steps of the model to collect transitions for before learning
    #: starts
    learning_starts: int = 100
    #: Minibatch size
    batch_size: Optional[int] = 64
    #: the soft update coefficient ("Polyak update", between 0 and 1)
    tau: float = 0.005
    #: Discount factor
    gamma: float = 0.99
    #: Enable a memory efficient variant of the replay buffer at a cost of more
    #: complexity. See
    #: https://github.com/DLR-RM/stable-baselines3/issues/37#:issuecomment
    # -637501195
    optimize_memory_usage: float = 0.95
    #: Seed for the pseudo random generators
    seed: Optional[int] = None
    #: Device (cpu, cuda, …) on which the code should be run. Setting it to
    #: auto, the code will be run on the GPU if possible.
    device: str = "auto"

    def get_agent(self,
                  environment: GymEnvironment,
                  normalizer_divisor: int = 1) -> DDPG:
        """
        Creates the stable_baselines version of the wanted Agent. Uses all
        Variables given in the object (except type) as the input parameters of
        the agent object creation.

        Args:
            environment (GymEnvironment):
                The environment the agent uses to train.
            normalizer_divisor (int):
                Currently not used in this function.

        Returns:
            DDPG: Initialized DDPG agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        return DDPG(env=environment, **self.get_additional_kwargs())


class SACAgent(BaseTrainingAgent):
    """
    SAC definition for the stable_baselines3 implementation for this algorithm.
    Variable comments are taken from the stable_baselines3 documentation.
    """
    #: What RL algorithm was used to train the agent. Needs to be know to
    #: correctly load the agent.
    type: Literal["SAC"]
    #: The learning rate, it can be a function of the current progress
    #: remaining (from 1 to 0)
    learning_rate: float = 1e-3
    #: size of the replay buffer
    buffer_size: int = 1000000
    #: how many steps of the model to collect transitions for before learning
    #: starts
    learning_starts: int = 100
    #: Minibatch size
    batch_size: Optional[int] = 64
    #: the soft update coefficient ("Polyak update", between 0 and 1)
    tau: float = 0.005
    #: Discount factor
    gamma: float = 0.99
    #: Enable a memory efficient variant of the replay buffer at a cost of more
    #: complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#:
    # issuecomment-637501195
    optimize_memory_usage: bool = False
    #: Entropy regularization coefficient. (Equivalent to inverse of reward
    #: scale in the original SAC paper.)  Controlling exploration/exploitation
    #: trade-off. Set it to 'auto' to learn it automatically (and 'auto_0.1'
    #: for using 0.1 as initial value)
    ent_coef: Union[str, float] = "auto"
    #: update the target network every ``target_network_update_freq`` gradient
    #: steps.
    target_update_interval: int = 1
    #: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    target_entropy: Union[str, float] = "auto"
    #: Whether to use generalized State Dependent Exploration (gSDE) instead
    #: of action noise exploration (default: False)
    use_sde: bool = False
    #: Sample a new noise matrix every n steps when using gSDE Default: -1
    #: (only sample at the beginning of the rollout)
    sde_sample_freq: int = -1
    #: Whether to use gSDE instead of uniform sampling during the warm up
    #: phase (before learning starts)
    use_sde_at_warmup: bool = False
    #: Seed for the pseudo random generators
    seed: Optional[int] = None
    #: Device (cpu, cuda, …) on which the code should be run. Setting it to
    #: auto, the code will be run on the GPU if possible.
    device: str = "auto"

    def get_agent(self,
                  environment: GymEnvironment,
                  normalizer_divisor: int = 1) -> SAC:
        """
        Creates the stable_baselines version of the wanted Agent. Uses all
        Variables given in the object (except type) as the input parameters of
        the agent object creation.

        Args:
            environment (GymEnvironment):
                The environment the agent uses to train.
            normalizer_divisor (int):
                Currently not used in this function.

        Returns:
            SAC: Initialized SAC agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        return SAC(env=environment, **self.get_additional_kwargs())


class DQNAgent(BaseTrainingAgent):
    """
    DQN definition for the stable_baselines3 implementation for this algorithm.
    Variable comments are taken from the stable_baselines3 documentation.
    """
    #: What RL algorithm was used to train the agent. Needs to be know to
    #: correctly load the agent.
    type: Literal["DQN"]
    #: The learning rate, it can be a function of the current progress
    #: remaining (from 1 to 0)
    learning_rate: float = 1e-4
    #: size of the replay buffer
    buffer_size: int = 1000000
    #: how many steps of the model to collect transitions for before learning
    #: starts
    learning_starts: int = 256
    #: Minibatch size for each gradient update
    batch_size: Optional[int] = 32
    #: the soft update coefficient ("Polyak update", between 0 and 1) default
    #: 1 for hard update
    tau: float = 1.0
    #: the discount factor
    gamma: float = 0.99
    #: Update the model every ``train_freq`` steps.
    train_freq: int = 4
    #: How many gradient steps to do after each rollout (see ``train_freq``)
    #: Set to ``-1`` means to do as many gradient steps as steps done in the
    #: environment during the rollout.
    gradient_steps: int = 1
    #: Enable a memory efficient variant of the replay buffer at a cost of more
    #: complexity. See https://github.com/DLR-RM/stable-baselines3/issues/37#:
    # issuecomment-637501195
    optimize_memory_usage: bool = False
    #: update the target network every ``target_update_interval``
    target_update_interval: int = 256
    #: fraction of entire training period over which the exploration rate is
    #: reduced
    exploration_fraction: float = 0.1
    #: initial value of random action probability
    exploration_initial_eps: float = 1.0
    #: final value of random action probability
    exploration_final_eps: float = 0.05
    #: The maximum value for the gradient clipping
    max_grad_norm: float = 10
    #: Seed for the pseudo random generators
    seed: Optional[int] = None
    #: Device (cpu, cuda, …) on which the code should be run. Setting it to
    #: auto, the code will be run on the GPU if possible.
    device: str = "auto"

    def get_agent(self,
                  environment: GymEnvironment,
                  normalizer_divisor: int = 1) -> DQN:
        """
        Creates the stable_baselines version of the wanted Agent. Uses all
        Variables given in the object (except type) as the input parameters of
        the agent object creation.

        Args:
            environment (GymEnvironment):
                The environment the agent uses to train.
            normalizer_divisor (int):
                Currently not used in this function.

        Returns:
            DQN: Initialized DQN agent.
        """
        self.get_logger().info(f"Using agent of type {self.type}.")
        return DQN(env=environment, **self.get_additional_kwargs())


AgentTypeDefinition = Union[PPOAgent, DDPGAgent, SACAgent, PretrainedAgent,
                            DQNAgent, A2CAgent]
