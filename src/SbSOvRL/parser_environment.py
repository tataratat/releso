from pydantic import BaseModel, conint
from typing import Optional, Any, List
# import gym
from SbSOvRL.spline import Spline
from SbSOvRL.mesh import Mesh
from SbSOvRL.solver import SolverTypeDefinition
from gustav import FreeFormDeformation
from pydantic.fields import PrivateAttr
from SbSOvRL.reward_parser import RewardFunctionTypes



class MultiProcessing(BaseModel):
    number_of_cores: conint(ge=1)


class Environment(BaseModel): # TODO make this gym environment
    multi_processing: Optional[MultiProcessing] = None
    spline: Spline
    mesh: Mesh
    solver: SolverTypeDefinition
    reward: RewardFunctionTypes
    
    # object variables
    _actions: List[Any] = PrivateAttr()
    _FFD: FreeFormDeformation = PrivateAttr(default_factory=FreeFormDeformation)

    #### object functions

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._actions = self.spline.get_actions()
        self._FFD.set_mesh(self.mesh.get_mesh())


    def is_multiprocessing(self):
        if self.multi_processing is None:
            return 0
        return self.multi_processing.number_of_cores

    def step(self):
        self._FFD.set_deformed_spline(self.spline.get_spline())
        self._FFD.deform_mesh()
