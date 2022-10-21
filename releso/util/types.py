"""Types and Type Unions defined for the ReLeSO package.

This file contains type definitions of commonly used items in this
framework/package. Mostly RL based here. Some more types can be found in other
files. Here only types with strictly external contents are created.
"""

from typing import Dict, Tuple, Union

from numpy import ndarray

#: Type definition for Observations.
ObservationType = Union[ndarray, Dict[str, ndarray], Tuple[ndarray, ...]]
#: Type definition for the info object returned for every step in the RL
#: Environment
InfoType = Dict
#: Type definition of the reward for every step in the RL Environment
RewardType = float
#: Type definition of the done marker for every step in the RL Environment
DoneType = bool

#: Type definition for the return value of a Step. A step return the steps
#: observation, generated reward, whether or not the current step is complete
#: and additional info.
StepReturnType = Tuple[ObservationType, RewardType, DoneType, InfoType]
