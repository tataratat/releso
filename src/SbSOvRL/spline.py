from abc import ABCMeta, abstractmethod
from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.util.logger import set_up_logger
from gustav import BSpline
from pydantic import BaseModel
from typing import List, Union, Optional, Dict, Any
import numpy as np
from pydantic.class_validators import root_validator, validator
from pydantic.fields import PrivateAttr
from pydantic.types import confloat, conint

parser_logger = set_up_logger("SbSOvRL_parser")
environment_logger = set_up_logger("SbSOvRL_environment")
 
class VariableLocation(BaseModel):
    current_position: confloat(le=1, ge=0)
    min_value: Optional[confloat(le=1, ge=0)] = None
    max_value: Optional[confloat(le=1, ge=0)] = None
    n_steps: Optional[conint(ge=1)] = 10
    step: Optional[confloat(le=1, ge=0)] = None
    
    # non json variables
    _is_action: Optional[bool] = PrivateAttr(default=None)

    @validator("min_value", "max_value", always=True)
    @classmethod
    def set_variable_to_current_position_if_not_given(cls, v, values, field) -> float:
        if v is None:
            if "current_position" in values.keys():
                return values["current_position"]
            else:
                raise SbSOvRLParserException("VariableLocation", field, "Please correctly define the current position.")
        return v

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
    @classmethod
    def define_step(cls, values: Dict[str, Any]) -> Dict[str, Any]:
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

    def is_action(self) -> bool:
        if self._is_action is None:
            self._is_action = (self.max_value > self.current_position) or (self.min_value < self.current_position)
        return self._is_action

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
    knot_vector: Optional[List[float]]

    @validator("knot_vector", always=True)
    @classmethod
    def validate_knot_vector(cls, v: Optional[List[float]], values: Dict[str, Any]) -> List[float]:
        if "number_of_points" in values.keys() and "degree" in values.keys() and "name" in values.keys():
            n_knots = values["number_of_points"] + values["degree"] + 1
        else:
            raise SbSOvRLParserException("SplineSpaceDimension", "knot_vector", "During validation the prerequisite variables number_of_points and degree were not present.")
        if type(v) is list:
            parser_logger.debug(f"The knot_vector for dimension {values['name']} is given.")
            if len(v) == n_knots:
                return v
            else:
                raise SbSOvRLParserException("SplineSpaceDimension", "knot_vector", f"The knot vector does not contain the correct number of items (is: {len(v)}, should_be: {n_knots}).")
        elif v is None:
            parser_logger.debug(f"The knot_vector for dimension {values['name']} is not given, trying to generate one in the open format.")
            parser_logger
            starting_ending = values["degree"] + 1
            middle = n_knots - (2 * starting_ending)
            if middle >= 0:
                knot_vec = list(np.array([np.zeros(starting_ending-1), np.linspace(0, 1, middle+2), np.ones(starting_ending-1)]).flatten())
            else:
                parser_logger.warning(f"The knot vector is shorter {n_knots} than the length given by the open format {starting_ending*2}. Knot vector is created by adding the starting and ending parts. The knot vector might be to long.")
                knot_vec = list(np.array([np.zeros(starting_ending), np.ones(starting_ending)]).flatten())
            return knot_vec

    def get_knot_vector(self) -> List[float]:
        return self.knot_vector

class SplineDefinition(BaseModel):
    space_dimensions: List[SplineSpaceDimension]
    spline_dimension: conint(ge=1)
    number_of_element_variables: conint(ge=0)
    control_point_variables: List[List[Union[VariableLocation, confloat(ge=0, le=1)]]]
    weights: Optional[List[float]] = [] # TODO check if correct number of weights are given

    @validator("control_point_variables", each_item=True)
    @classmethod
    def convert_all_control_point_locations_to_variable_locations(cls, v):
        new_list = []
        for element in v:
            new_list.append(VariableLocation(current_position=element) if type(element) is float else element)
        return new_list

    @validator("weights", always=True)
    @classmethod
    def validate_weights(cls, v: Optional[List[float]], values: Dict[str, Any]) -> List[float]:
        """Validate if the correct number of weights are present in the weight vector. If weight vector is not given weight vector should be all ones.

        Args:
            v (Optional[List[float]]): Value to validate
            values (Dict[str, Any]): Previously validated variables

        Raises:
            SbSOvRLParserException: Parser Error

        Returns:
            List[float]: Filled weight vector.
        """
        if "space_dimensions" not in values.keys():
            raise SbSOvRLParserException("SplineDefinition", "weights", "During validation the prerequisite variable space_dimensions were not present.")
        n_cp = np.prod([space_dim.number_of_points for space_dim in values["space_dimensions"]])
        if type(v) is list:
            if len(v) == n_cp:
                parser_logger.debug("Found correct number of weights in SplineDefinition.")
            else:
                raise SbSOvRLParserException("SplineDefinition", "weights", f"The length of the weight vector {len(v)} is not the same as the number of control_points {n_cp}.")
        elif v is None:
            v = list(np.ones(n_cp))
        return v

    def get_number_of_points(self) -> int:
        """Returns the number of points in the Spline. Currently the number of points in the spline is calculated by multiplying the number of points in each spline dimension.

        Returns:
            int: number of points in the spline
        """
        return np.prod([dimension.number_of_points for dimension in self.space_dimensions])

    def get_controll_points(self):
        return [[controll_point.current_position for controll_point in sub_list] for sub_list in self.control_point_variables]

    def get_actions(self) -> List[VariableLocation]:
        """Returns list of VariableLocations but only if the variable location is actually variable.

        Returns:
            List[VariableLocation]: See above
        """
        environment_logger.debug("Collecting all actions for BSpline.")
        return [variable for sub_dim in self.control_point_variables for variable in sub_dim if variable.is_action()]

    @abstractmethod
    def get_spline(self) -> Any:
        """Generates the current Spline in the gustav format.

        Notes:
            This is an abstract method.

        Returns:
            Spline: Spline that is generated.
        """
        raise NotImplementedError

class BSplineDefinition(SplineDefinition):

    def get_spline(self) -> BSpline:
        """Creates the BSpline from the defintion given by the json file.

        Returns:
            BSpline: given by the #degrees and knot_vector in each space_dimension and the current controll points.
        """
        environment_logger.debug("Creating Gustav BSpline.")
        environment_logger.debug(f"degree vector: {[space_dim.degree for space_dim in self.space_dimensions]}")
        environment_logger.debug(f"knot vector: {[space_dim.get_knot_vector() for space_dim in self.space_dimensions]}")
        environment_logger.debug(f"controll_points: {self.get_controll_points()}")
        
        return BSpline(
            [space_dim.degree for space_dim in self.space_dimensions],
            [space_dim.get_knot_vector() for space_dim in self.space_dimensions],
            self.get_controll_points()
            )

SplineTypes = Union[BSplineDefinition] # should always be a derivate of SplineDefinition #TODO add NURBS

class Spline(BaseModel):
    spline_definition: SplineTypes

    def get_spline(self) -> Union[BSpline]:
        """Creates the current spline.

        Returns:
            Union[BSpline]: [description]
        """
        environment_logger.debug("Getting Gustav Spline.")
        return self.spline_definition.get_spline()

    def get_actions(self) -> List[VariableLocation]:
        """Returns a list of VariableLocations that actually have a defined variability and can therefor be an action.

        Returns:
            List[VariableLocation]: VariableLocations that have wriggle room.
        """
        return self.spline_definition.get_actions()
    
