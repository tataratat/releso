"""This file contains type definitions of commoly used items in this framework/package. 
"""

from typing import Dict, Tuple, Union
from numpy import ndarray

ObservationType = Union[ndarray, Dict[str, ndarray], Tuple[ndarray, ...]]  #: Type definition for Observations.
InfoType = Dict  #: Type definition for the info object returned for every step in the RL Environment
RewardType = float #: Type definition of the reward for every step in the RL Environment
DoneType = bool  #: Type definition of the done marker for every step in the RL Environment 

StepReturnType = Tuple[ObservationType, RewardType, DoneType, InfoType]  #: Type definition for the return value of a Step. A step return the steps observation, generated reward, whether or not the current step is complete and additional info.
