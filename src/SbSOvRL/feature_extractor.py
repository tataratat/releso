"""
File defines the FeatureExtractors currently implemented in this framework.
These are implemented in a more or less slapdash fashion. Use at own Risk.
"""
import logging
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
import torch as th
from gym import Space, spaces
from typing import Dict, Any, Literal, Optional
import numpy as np
from torchvision import models, transforms
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.preprocessing import is_image_space


class SbSOvRL_FeatureExtractor(BaseFeaturesExtractor):
    """
    The standard feature extractor can only be used with a Box defined
    Observation space. For more complex Observations spaces see
    :py:class:`SbSOvRL.feature_extractor.SbSOvRL_CombinedExtractor`
    """

    def __init__(
            self, observation_space: Space, features_dim=512,
            without_linear: bool = False,
            network_type: Literal["resnet18", "mobilenetv2"] = "resnet18",
            logger: Optional[logging.Logger] = None):
        """
        The feature extractor is used to also be able to handle image based
        observations better and to give the option to used pretrained networks
        as the feature extractor. Multiple are available please look into the
        code to check the currently available pretrained networks. Not all are
        listed in the network_type variable hint.

        Args:
          observation_space (Space):
            Observations space of the environment.
          cnn_output_dim (int, optional):
            How many features the feature extractor should return.
            Defaults to 256.
          without_linear (bool, optional):
            Use the pretrained nets without a linear layer at the end. Setting
            the output_dim is not possible since the output sie is directly
            dependent on the input size. Defaults to False.
          network_type (Literal["resnet18", "mobilenetv2"], optional):
            Network to use for the cnn extractor. Defaults to "resnet18".
          logger (Optional[logging.Logger], optional):
            Logger to use for logging purposes. Defaults to None.

        """
        super().__init__(observation_space, features_dim=features_dim)
        self.without_linear = without_linear
        # Define the network
        if network_type == "resnet18":
            pre_network = models.resnet18(pretrained=True)
            self.model = th.nn.Sequential(*[
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                pre_network.conv1, pre_network.bn1, pre_network.relu,
                pre_network.maxpool, pre_network.layer1, pre_network.layer2,
                pre_network.layer3, pre_network.layer4, th.nn.Flatten()])
        elif network_type == "mobilenetv2":
            pre_network = models.mobilenet_v2(pretrained=True)
            self.model = th.nn.Sequential(*[
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                pre_network.features, th.nn.Flatten()])
        elif network_type == "mobilenetv3_small":
            pre_network = models.mobilenet_v3_small(pretrained=True)
            self.model = th.nn.Sequential(*[
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                pre_network.features, th.nn.Flatten()])
        elif network_type == "mobilenetv3_large":
            pre_network = models.mobilenet_v3_large(pretrained=True)
            self.model = th.nn.Sequential(*[
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
                pre_network.features, th.nn.Flatten()])
        elif network_type == "inception_v3":
            if self.without_linear:
                if logger:
                    logger.error(
                        "FeatureExtractor: Inception_v3 can only be used with "
                        "linear layer.")
                raise RuntimeError(
                    "FeatureExtractor: Inception_v3 can only be used with "
                    "linear layer.")
            self.model = models.inception_v3(pretrained=True)
        else:
            if logger:
                logger.error(
                    f"The given network type of {network_type} is unknown. "
                    "Please choose one that is known.")
            raise RuntimeError(
                f"FeatureExtractor: Given network type -{network_type}- "
                "unknown.")
        # deactivate training
        for param in self.model.parameters():
            param.requires_grad = False

        # # Define the feature dimension of the network
        # gen = np.random.default_rng()
        # ins = gen.random((64,*observation_space.shape))
        # ins_torch = from_numpy(ins).float().to("cpu")
        # self._features_dim = sum(self.model(ins_torch).shape[1:])

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.model(th.as_tensor(
                observation_space.sample()[None]).float()).shape
        if self.without_linear:
            features_dim = n_flatten[1]
        else:
            if network_type == "inception_v3":
                self.model.fc = th.nn.Linear(
                    self.model.fc.in_features, features_dim, th.nn.ReLU())
                # only so that forward correctly works. Actually uses the
                # linear layer but it is included inside the model itself.
                self.without_linear = True
            else:
                self.linear = th.nn.Sequential(th.nn.Linear(
                    n_flatten[1], features_dim), th.nn.ReLU())
        if logger:
            logger.warning(
                f"FeatureExtractor: Used pretrained {network_type} results in "
                f"{features_dim} feature dims.")
        self._features_dim = features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Forward pass of the Network defined.

        Args:
            observations (th.Tensor): Input for the network

        Returns:
            th.Tensor: Output of the network
        """
        # print("-"*20,flush=True)
        # print(observations.shape)
        # print(th.min(observations))
        # print(th.max(observations))
        # print("-"*20,flush=True)
        observations = self.model(observations)
        # return observations
        if not self.without_linear:
            observations = self.linear(observations)
        return observations


class SbSOvRL_CombinedExtractor(BaseFeaturesExtractor):
    """
      Combined Extractor can use a Dict definition of the observations space.

      Notes:
        Class is a direct copy from
        stable_baselines3.common.torch_layers.CombinedExtractor. Only change is
        the image feature extractor.
    """

    def __init__(
      self, observation_space: spaces.Dict, cnn_output_dim: int = 256,
      without_linear: bool = False,
      network_type: Literal["resnet18", "mobilenetv2"] = "resnet18",
      logger: Optional[logging.Logger] = None):
        """
          The feature extractor is used to also be able to handle image based
          observations better and to give the option to used pretrained
          networks as the feature extractor. Multiple are available please look
          into the code to check the currently available pretrained networks.
          Not all are listed in the network_type variable hint.

          The combined feature extractor extends this capability to non uniform
          observations space definitions. For example if image based
          observations are mixed with standard (scalar) observations. Or
          multiple images are used as an observation.

        Args:
            observation_space (spaces.Dict):
              Observations space of the environment.
            cnn_output_dim (int, optional):
              How many features the feature extractor should return.
              Defaults to 256.
            without_linear (bool, optional):
              Use the pretrained nets without a linear layer at the end.
              Setting the output_dim is not possible since the output sie is
              directly dependent on the input size. Defaults to False.
            network_type (Literal["resnet18", "mobilenetv2"], optional):
              Network to use for the cnn extractor. Defaults to "resnet18".
            logger (Optional[logging.Logger], optional):
              Logger to use for logging purposes. Defaults to None.
        """
        # TODO we do not know features-dim here before going over all the
        # items, so put something there. This is not clean!
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = SbSOvRL_FeatureExtractor(
                    observation_space=subspace, features_dim=cnn_output_dim,
                    without_linear=without_linear, network_type=network_type,
                    logger=logger)
                total_concat_size += extractors[key]._features_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = th.nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = th.nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        """Forward pass of the Network defined.

        Args:
            observations (th.Tensor): Input for the network

        Returns:
            th.Tensor: Output of the network
        """
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
