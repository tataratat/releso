"""Definition of Observation space definitions.

These are used to parse and handle the observations inside the package. The
biggest functional change is that they provide default observations. Needed
in cases where observations are depended on a previous step which errored out
and could not be completed.
"""

from typing import List, Literal, Tuple

import numpy as np
from gymnasium.spaces import Box, Space

from releso.base_model import BaseModel


class ObservationDefinition(BaseModel):
    """Definition of an Observation.

    Definition of a single Observations by providing the name of the
    value and the range in which the value is limited to.

    The range is necessary due to normalization of the input of the agent
    networks.
    """

    name: str  #: Name of the observation
    #: minimum of the range in which the observation is bound
    value_min: float
    #: maximum of the range in which the observation is bound
    value_max: float

    def get_observation_definition(self) -> Tuple[str, Space]:
        """Provide definition of the defined observations space.

        Returns a tuple of name and observation definition defined via the
        gymnasium.observation interface

        Returns:
            Tuple[str, Space]: Tuple of the name of the observation and a
            gymnasium.Box definition
        """
        return (
            self.name,
            Box(self.value_min, self.value_max, shape=([1]), dtype=np.float32),
        )

    def get_default_observation(self) -> np.ndarray:
        """Provide default observations.

        Gives a default observations of the correct shape, purpose is when the
        observation fails that the observation can still be generated so that
        the training does not error out.

        Returns:
            np.ndarray: An array filled with ones in the correct shape and size
            of the observation.
        """
        return np.ones((1,)) * self.value_min


class ObservationDefinitionMulti(ObservationDefinition):
    """Define a multidimensional Observations space.

    Definition of a single Observations by providing the name of the value
    and the range in which the value is limited to.

    The range is necessary due to normalization of the input of the agent
    networks.
    """

    #: Shape of the Observation space. List of number of elements per dimension
    observation_shape: List[int]
    #: Type of the Observation space. If float uses the value_min etc
    #: definition for the limits of the space. If CNN uses [0, 255] limits.
    value_type: Literal["float", "CNN"]

    def get_observation_definition(self) -> Tuple[str, Space]:
        """Provide definition of the defined observations space.

        Returns a tuple of name and observation definition defined via the
        gymnasium.observation interface

        Returns:
            Tuple[str, Space]: Tuple of the name of the observation
                and a gymnasium.Box definition
        """
        if self.value_type == "CNN":
            return (
                self.name,
                Box(
                    low=0,
                    high=255,
                    shape=self.observation_shape,
                    dtype=np.uint8,
                ),
            )
        return (
            self.name,
            Box(
                self.value_min,
                self.value_max,
                shape=(self.observation_shape),
                dtype=np.float32,
            ),
        )

    def get_default_observation(self) -> np.ndarray:
        """Provide default observations.

        Gives a default observations of the correct shape, purpose is when the
        observation fails that the observation can still be generated so that
        the training does not error out.

        Returns:
            np.ndarray: An array filled with ones in the correct shape and size
            of the observation.
        """
        if self.value_type == "CNN":
            return np.zeros(self.observation_shape, dtype=np.uint8)
        return (
            np.ones(self.observation_shape, dtype=np.float32) * self.value_min
        )
