from SbSOvRL.exceptions import SbSOvRLParserException
from pydantic import BaseModel
from typing import Literal, List, Union, Optional, Dict, Any
import numpy as np
from pydantic.class_validators import root_validator, validator
from pydantic.types import confloat, conlist, conint
 
class VariableLocation(BaseModel):
    current_position: confloat(le=1, ge=0)
    min_value: Optional[confloat(le=1, ge=0)] = None
    max_value: Optional[confloat(le=1, ge=0)] = None
    discrete: bool = False
    n_steps: Optional[conint(ge=1)] = 10
    step: Optional[confloat(le=1, ge=0)] = None

    @validator("min_value", "max_value", always=True)
    @classmethod
    def set_variable_to_current_position_if_not_given(cls, v, values, field) -> float:
        if v is None:
            if "current_position" in values.keys():
                return values["current_position"]
            else:
                raise SbSOvRLParserException("VariableLocation", field, "Please correctly define the current position.")
        return v

    # @validator("max_value", always=True)
    # @classmethod
    # def max_value_is_greater_than_min_value(cls, v, values, field) -> float:
    #     if "min_value" not in values.keys():
    #         raise SbSOvRLParserException("VariableLocation", field, "Please define the min_value.")
    #     if v is None:
    #         raise RuntimeError("This should not have happended.")
    #     if v<values["min_value"]:
    #         raise SbSOvRLParserException("VariableLocation", field, f"The min_value {values['min_value']} must be smaller or equal to the max_value {v}.")

    @validator("max_value", always=True)
    @classmethod
    def max_value_is_greater_than_min_value(cls, v, values, field) -> float:
        if "min_value" not in values.keys():
            raise SbSOvRLParserException("VariableLocation", field, "Please define the min_value.")
        if v is None:
            raise RuntimeError("This should not have happended.")
        if v < values["min_value"]:
            raise SbSOvRLParserException("VariableLocation", field, f"The min_value {values['min_value']} must be smaller or equal to the max_value {v}.")
        return v

    @root_validator
    def define_step(cls, values) -> Dict[str, Any]:
        discrete = values.get("discrete")
        if discrete:
            step, n_steps, min_value, max_value = values.get("step"), values.get("n_steps"), values.get("min_value"), values.get("max_value")
            value_range = max_value - min_value
            if step is not None:
                return values # if step length is defined ignore n_steps
            elif n_steps is None:
                n_steps = 10 # if discrete but neither step nor n_steps is defined a default 10 steps are assumed.
            step = value_range/n_steps
            values["step"] = step
        return values

    def is_action_discret(self) -> bool:
        return self.discrete

    def apply_discrete_action(self, increasing: bool) -> float:
        step = self.step if increasing else - self.step
        self.current_position = np.clip(self.current_position + step, self.min_value, self.max_value)
        return self.current_position

    def apply_continuos_action(self, value: float) -> float:
        self.current_position = np.clip(self.current_position + value, self.min_value, self.max_value)
        return self.current_position

class SplineSpaceDimension(BaseModel):
    name: str
    number_of_points: conint(ge=1)
    degree: conint(ge=1)
    knot_vector: List[float]


class SplineDefinition(BaseModel):
    space_dimensions: List[SplineSpaceDimension]
    spline_dimension: conint(ge=1)
    number_of_element_variables: conint(ge=0)
    control_point_variables: List[List[Union[VariableLocation, confloat(ge=0, le=1)]]]
    weights: List[float] = [] # TODO check if correct number of weights are given

    @validator("control_point_variables", each_item=True)
    @classmethod
    def convert_all_control_point_locations_to_variable_locations(cls, v):
        new_list = []
        for element in v:
            new_list.append(VariableLocation(current_position=element) if type(element) is float else element)
        return new_list

    def get_number_of_points(self) -> int:
        """Returns the number of points in the Spline. Currently the number of points in the spline is calculated by multiplying the number of points in each spline dimension.

        Returns:
            int: number of points in the spline
        """
        return np.prod([dimension.number_of_points for dimension in self.space_dimensions])

class Spline(BaseModel):
    spline_type: Literal[1]
    spline_definition: SplineDefinition
    

SplineTypeDefinition = Union[Spline]