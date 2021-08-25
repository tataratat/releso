from pydantic import BaseModel, conint
from typing import Optional
# from SbSOvRL.parser_functions import validate_file_path_to_absolute
# import gym
from SbSOvRL.spline import SplineTypeDefinition
from SbSOvRL.mesh import Mesh
from SbSOvRL.solver import SolverTypeDefinition



class MultiProcessing(BaseModel):
    number_of_cores: conint(ge=1)


class Environment(BaseModel):
    multi_processing: Optional[MultiProcessing] = None
    spline: SplineTypeDefinition
    mesh: Mesh
    solver: SolverTypeDefinition
    
    #### object functions

    def is_multiprocessing(self):
        if self.multi_processing is None:
            return 0
        return self.multi_processing.number_of_cores