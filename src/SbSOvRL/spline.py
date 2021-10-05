from abc import abstractmethod
from SbSOvRL.exceptions import SbSOvRLParserException
import logging
from gustav import BSpline, NURBS
from SbSOvRL.base_model import SbSOvRL_BaseModel
from typing import List, Union, Optional, Dict, Any
import numpy as np
from pydantic.class_validators import root_validator, validator
from pydantic.fields import PrivateAttr
from pydantic.types import confloat, conint
import copy
 
class VariableLocation(SbSOvRL_BaseModel):
    current_position: float
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    n_steps: Optional[conint(ge=1)] = 10
    step: Optional[float] = None
    
    # non json variables
    _is_action: Optional[bool] = PrivateAttr(default=None)
    _original_position: Optional[float] = PrivateAttr(default=None)

    @validator("min_value", "max_value", always=True)
    @classmethod
    def set_variable_to_current_position_if_not_given(cls, v, values, field) -> float:
        """Validation of the min and max values for the current VariableLocation. If non are set no variability is assumed min = max = current_position

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            SbSOvRLParserException: Parser error if current_position is not already validated.

        Returns:
            float: value of the validated value.
        """
        if v is None:
            if "current_position" in values.keys():
                return values["current_position"]
            else:
                raise SbSOvRLParserException("VariableLocation", field, "Please correctly define the current position.")
        return v

    @validator("max_value", always=True)
    @classmethod
    def max_value_is_greater_than_min_value(cls, v, values, field) -> float:
        """Validates that the max value is greater or equal to the min value.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            SbSOvRLParserException: Parser error if min_value is not already validated and if min value greater if max value.

        Returns:
            float: value of the validated value.
        """
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
        """Validation function that defines the step taken if the action would be discrete. If nothing is given for the calculation a n_steps default value of 10 is used.

        Args:
            values (Dict[str, Any]): Validated variables of the object.

        Returns:
            Dict[str, Any]: Validated variables but if steps was not given before now it has a value.
        """
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
        """Checks if the variable has variability. Meaning is there optimization between the min_value and the max_value.

        Returns:
            bool: True if (self.max_value > self.current_position) or (self.min_value < self.current_position)
        """
        if self._is_action is None:
            self._is_action = (self.max_value > self.current_position) or (self.min_value < self.current_position)
        return self._is_action

    def apply_discrete_action(self, increasing: bool) -> float:
        """Apply the a discrete action to the current value. Also applies clipping if min/max value is surpassed.

        Args:
            increasing (bool): Toggle whether or not the value should increase or not.

        Returns:
            float: Value of the position that the current position is now at.
        """
        if self._original_position is None:
            self._original_position = self.current_position
        step = self.step if increasing else -self.step
        # logging.getLogger("SbSOvRL_environment").debug(f"Current step is {step} while increasing: {increasing}")
        self.current_position = np.clip(self.current_position + step, self.min_value, self.max_value)
        return self.current_position

    def apply_continuos_action(self, value: float) -> float:
        """Apply the zero-mean normalized value as the new current position. Needs to descale the value first. Also applies clipping.
        Args:
            value (float): Scaled value [-1,1] that needs to be descaled and applied as the new current position.

        Returns:
            float: New clipped value.
        """
        if self._original_position is None:
            self._original_position = self.current_position
        delta = self.max_value - self.min_value
        descaled_value = ((value+1.)/2.) * delta
        self.current_position = np.clip(descaled_value, self.min_value, self.max_value)
        return self.current_position

    def reset(self) -> None:
        """Resets the current_location to the initial position.
        """
        if self._original_position is not None:
            self.current_position = self._original_position

class SplineSpaceDimension(SbSOvRL_BaseModel):
    name: str
    number_of_points: conint(ge=1)
    degree: conint(ge=1)
    knot_vector: Optional[List[float]]

    @validator("knot_vector", always=True)
    @classmethod
    def validate_knot_vector(cls, v: Optional[List[float]], values: Dict[str, Any]) -> List[float]:
        """If knot vector not given tries to make a default open knot vector. If knot vector is 

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            SbSOvRLParserException: Emitted parser error.

        Returns:
            float: value of the validated value.
        """
        if "number_of_points" in values.keys() and "degree" in values.keys() and "name" in values.keys():
            n_knots = values["number_of_points"] + values["degree"] + 1
        else:
            raise SbSOvRLParserException("SplineSpaceDimension", "knot_vector", "During validation the prerequisite variables number_of_points and degree were not present.")
        if type(v) is list:
            logging.getLogger("SbSOVRL_parser").debug(f"The knot_vector for dimension {values['name']} is given.")
            if len(v) == n_knots:
                return v
            else:
                logging.getLogger("SbSOVRL_parser").warning(f"The knot vector does not contain the correct number of items for an open knot vector definition. (is: {len(v)}, should_be: {n_knots}).")
        elif v is None:
            logging.getLogger("SbSOVRL_parser").debug(f"The knot_vector for dimension {values['name']} is not given, trying to generate one in the open format.")
            starting_ending = values["degree"] + 1
            middle = n_knots - (2 * starting_ending)
            if middle >= 0:
                knot_vec = list(np.array([np.zeros(starting_ending-1), np.linspace(0, 1, middle+2), np.ones(starting_ending-1)]).flatten())
            else:
                logging.getLogger("SbSOVRL_parser").warning(f"The knot vector is shorter {n_knots} than the length given by the open format {starting_ending*2}. Knot vector is created by adding the starting and ending parts. The knot vector might be to long.")
                knot_vec = list(np.array([np.zeros(starting_ending), np.ones(starting_ending)]).flatten())
            return knot_vec

    def get_knot_vector(self) -> List[float]:
        """Function returns the Knot vector given in the object.

        Returns:
            List[float]: Direct return of the inner variable.
        """
        return self.knot_vector

class SplineDefinition(SbSOvRL_BaseModel):
    space_dimensions: List[SplineSpaceDimension]
    spline_dimension: conint(ge=1)
    control_point_variables: Optional[List[List[Union[VariableLocation, confloat(ge=0, le=1)]]]]

    @validator("control_point_variables", always=True)
    @classmethod
    def make_default_controll_point_grid(cls, v, values) -> List[List[Union[VariableLocation]]]:
        """If value is None a equidistant grid of controll points will be given with variability of half the space between the each control point.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            SbSOvRLParserException: Emitted parser error.

        Returns:
            List[List[VariableLocation]]: Definition of the controll_points
        """
        # raise SbSOvRLParserException("SplineDefinition", "control_point_variables", "asdasdasd.")
        if v is None:
            if "space_dimensions" not in values.keys():
                raise SbSOvRLParserException("SplineDefinition", "control_point_variables", "During validation the prerequisite variable space_dimensions was not present.")
            spline_dimensions = values["space_dimensions"]
            n_points_in_dim = [dim.number_of_points for dim in spline_dimensions]
            # this can be done in a smaller and faster footprint but for readability and understanding 
            dimension_lists = []
            dim_spacings = []
            # create value range in each dimension separately
            for n_p in n_points_in_dim:
                dimension_lists.append(list(np.linspace(0.,1.,n_p)))
                dim_spacings.append(1./((n_p-1)*2))

            # create each controll point for by concatenating each value in each list with each value of all other lists
            for inner_dim, dim_spacing in zip(dimension_lists, dim_spacings):
                if v is None: # first iteration the v vector must be initialized with the first dimesion vector
                    v = []
                    for element in inner_dim:
                        if element == 0.:
                            v.append(VariableLocation(current_position = element, min_value = element, max_value = element+dim_spacing))
                        elif element == 1.:
                            v.append(VariableLocation(current_position = element, min_value = element-dim_spacing, max_value=element))
                        else:
                            v.append(VariableLocation(current_position = element, min_value = element-dim_spacing, max_value=element+dim_spacing))
                else: # for each successive dimension for each existing element in v each value in the new dimesion must be added
                    temp_v = []
                    for current_list in v:
                        for element in inner_dim:
                            elem = copy.deepcopy(current_list if type(current_list) is list else [current_list])
                            if element == 0.:
                                temp_v.append([VariableLocation(current_position = element, min_value = element, max_value = element+dim_spacing)] + elem)
                            elif element == 1.:
                                temp_v.append([VariableLocation(current_position = element, min_value = element-dim_spacing, max_value=element)] + elem)
                            else:
                                temp_v.append([VariableLocation(current_position = element, min_value = element-dim_spacing, max_value=element+dim_spacing)] + elem)
                    v = temp_v
        return v


    @validator("control_point_variables", each_item=True)
    @classmethod
    def convert_all_control_point_locations_to_variable_locations(cls, v):
        """ Converts all controll points values into VariabelLocations if the value is given as a simple float. Simple float will be converted into VariableLocation with current_position=value and no variability.

        Args:
            v ([type]): value to validate

        Returns:
            [type]: validated value
        """
        new_list = []
        for element in v:
            new_list.append(VariableLocation(current_position=element) if type(element) is float else element)
            if not 0. <= element.current_position <= 1.:
                raise SbSOvRLParserException("SplineDefinition", "controll_point_variables", "The controll_point_variables need to be inside an unit hypercube. Found a values outside this unit hypercube.")
        return new_list

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
        logging.getLogger("SbSOvRL_environment").debug("Collecting all actions for BSpline.")
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
    
    def reset(self) -> None:
        """Resets all control points in the spline to its original position (position they were initialized with).
        """
        for control_point in self.control_point_variables:
            for dim in control_point:
                dim.reset()

class BSplineDefinition(SplineDefinition):

    def get_spline(self) -> BSpline:
        """Creates the BSpline from the defintion given by the json file.

        Returns:
            BSpline: given by the #degrees and knot_vector in each space_dimension and the current controll points.
        """
        logging.getLogger("SbSOvRL_environment").debug("Creating Gustav BSpline.")
        # logging.getLogger("SbSOvRL_environment").debug(f"degree vector: {[space_dim.degree for space_dim in self.space_dimensions]}")
        # logging.getLogger("SbSOvRL_environment").debug(f"knot vector: {[space_dim.get_knot_vector() for space_dim in self.space_dimensions]}")
        logging.getLogger("SbSOvRL_environment").debug(f"With controll_points: {self.get_controll_points()}")
        
        return BSpline(
            [space_dim.degree for space_dim in self.space_dimensions],
            [space_dim.get_knot_vector() for space_dim in self.space_dimensions],
            self.get_controll_points()
            )

class NURBSDefinition(SplineDefinition):
    weights: List[Union[float, VariableLocation]] # TODO should weights also be changable?

    @validator("weights")
    @classmethod
    def validate_weights(cls, v: Optional[List[Union[float, VariableLocation]]], values: Dict[str, Any]) -> List[Union[float, VariableLocation]]:
        """Validate if the correct number of weights are present in the weight vector. If weight vector is not given weight vector should be all ones.

        Args:
            v (List[Union[float, VariableLocation]]): Value to validate
            values (Dict[str, Any]): Previously validated variables

        Raises:
            SbSOvRLParserException: Parser Error

        Returns:
            List[Union[float, VariableLocation]]: Filled weight vector.
        """
        if "space_dimensions" not in values.keys():
            raise SbSOvRLParserException("SplineDefinition", "weights", "During validation the prerequisite variable space_dimensions were not present.")
        n_cp = np.prod([space_dim.number_of_points for space_dim in values["space_dimensions"]])
        if type(v) is list:
            if len(v) == n_cp:
                logging.getLogger("SbSOVRL_parser").debug("Found correct number of weights in SplineDefinition.")
            else:
                raise SbSOvRLParserException("SplineDefinition NURBS", "weights", f"The length of the weight vector {len(v)} is not the same as the number of control_points {n_cp}.")
        elif v is None:
            v = list(np.ones(n_cp))
        return v

    @validator("weights", each_item=True)
    @classmethod
    def convert_weights_into_variable_location(cls, v: List[Union[float, VariableLocation]]) -> List[VariableLocation]:
        """Convert all float values in the weight vector into VariableLocations. So that these can also be used as actions.

        Args:
            v (List[Union[float, VariableLocation]]): Value to validate

        Returns:
            List[VariableLocation]: Filled weight vector.
        """
        if type(v) is float:
            return VariableLocation(current_position=v)
        return v

    def get_spline(self) -> NURBS:
        """Creates a NURBSSpline from the defintion given by the json file.

        Returns:
            NURBS: NURBSSpline
        """
        logging.getLogger("SbSOvRL_environment").debug("Creating Gustav NURBS.")
        # logging.getLogger("SbSOvRL_environment").debug(f"degree vector: {[space_dim.degree for space_dim in self.space_dimensions]}")
        # logging.getLogger("SbSOvRL_environment").debug(f"knot vector: {[space_dim.get_knot_vector() for space_dim in self.space_dimensions]}")
        # logging.getLogger("SbSOvRL_environment").debug(f"controll_points: {self.get_controll_points()}")
        
        return NURBS(
            [space_dim.degree for space_dim in self.space_dimensions],
            [space_dim.get_knot_vector() for space_dim in self.space_dimensions],
            self.get_controll_points(),
            self.weights
            )
    
    def get_actions(self) -> List[VariableLocation]:
        """Extends the controll point actions with the weight actions.

        Returns:
            List[VariableLocation]: List of possible actions for this NURB spline.
        """
        actions = super().get_actions()
        actions.extend([variable for variable in self.weights if variable.is_action()])
        return actions

    def reset(self) -> None:
        super().reset()
        for weight in self.weights:
            weight.reset()

SplineTypes = Union[NURBSDefinition, BSplineDefinition] # should always be a derivate of SplineDefinition

class Spline(SbSOvRL_BaseModel):
    spline_definition: SplineTypes

    def get_spline(self) -> Union[BSpline, NURBS]:
        """Creates the current spline.

        Returns:
            Union[BSpline, NURBS]: [description]
        """
        return self.spline_definition.get_spline()

    def get_actions(self) -> List[VariableLocation]:
        """Returns a list of VariableLocations that actually have a defined variability and can therefor be an action.

        Returns:
            List[VariableLocation]: VariableLocations that have 'wriggle' room.
        """
        return self.spline_definition.get_actions()

    def reset(self) -> None:
        """Resets the spline to its initial values.
        """
        self.spline_definition.reset()
    
