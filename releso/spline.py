"""Definition of the action space.

File holds all classes which define the spline and with that also the action
definition of the problem.
"""

import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic.class_validators import root_validator, validator
from pydantic.types import conint

from releso.base_model import BaseModel
from releso.exceptions import ParserException
from releso.shape_parameterization import ShapeDefinition, VariableLocation
from releso.util.logger import get_parser_logger

try:
    from splinepy import NURBS, BSpline
except ImportError as err:  # pragma: no cover
    from releso.util.module_import_raiser import ModuleImportRaiser

    BSpline = ModuleImportRaiser("splinepy", err)
    NURBS = ModuleImportRaiser("splinepy", err)


class SplineSpaceDimension(BaseModel):
    """Defines a single spline space dimension of the current spline.

    The dimension is a dimension of the parametric spline dimension.
    """

    #: Name of the spline dimension used for easier identification
    name: str
    #: number of control points in this dimension
    number_of_points: conint(ge=1)
    #: degree of the spline in this dimension
    degree: conint(ge=1)
    #: knot vector describing the knot intervals of the spline in the current
    #: dimension
    knot_vector: Optional[List[float]]

    @validator("knot_vector", always=True)
    @classmethod
    def validate_knot_vector(
        cls, v: Optional[List[float]], values: Dict[str, Any]
    ) -> List[float]:
        """Validator for knot_vector.

        If knot vector not given tries to make a default open knot vector.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            ParserException: Emitted parser error.

        Returns:
            float: value of the validated value.
        """
        if (
            "number_of_points" in values.keys()
            and "degree" in values.keys()
            and "name" in values.keys()
        ):
            n_knots = values["number_of_points"] + values["degree"] + 1
        else:
            raise ParserException(
                "SplineSpaceDimension",
                "knot_vector",
                "During validation the "
                "prerequisite variables number_of_points and degree were not "
                "present.",
            )
        if isinstance(v, list):
            get_parser_logger().debug(
                f"The knot_vector for dimension {values['name']} is given."
            )
            if len(v) == n_knots:
                return v
            else:
                raise ParserException(
                    "SplineSpaceDimension",
                    "knot_vector",
                    f"The knot vector does not contain the correct number of "
                    f"items for an open knot vector definition. (is: {len(v)},"
                    f" should_be: {n_knots}).",
                )
        elif v is None:
            get_parser_logger().debug(
                f"The knot_vector for dimension {values['name']} is not given,"
                f" trying to generate one in the open format."
            )
            starting_ending = values["degree"] + 1
            middle = n_knots - (2 * starting_ending)
            if middle >= 0:
                knot_vec = list(
                    np.append(
                        np.append(
                            np.zeros(starting_ending - 1),
                            np.linspace(0, 1, middle + 2),
                        ),
                        np.ones(starting_ending - 1),
                    )
                )
            else:
                get_parser_logger().warning(
                    f"The knot vector is shorter {n_knots} than the length "
                    f"given by the open format {starting_ending * 2}. Knot "
                    "vector is created by adding the starting and ending "
                    "parts. The knot vector might be to long."
                )
                knot_vec = list(
                    np.array([
                        np.zeros(starting_ending),
                        np.ones(starting_ending),
                    ]).flatten()
                )
            return knot_vec

    def get_knot_vector(self) -> List[float]:
        """Function returns the Knot vector given in the object.

        Returns:
            List[float]: Direct return of the inner variable.
        """
        return self.knot_vector


class SplineDefinition(ShapeDefinition):
    """Defines the spline.

    Base class for the NURBS and B-Spline implementations.
    """

    #: Definition of the space dimensions of the spline
    space_dimensions: List[SplineSpaceDimension]
    # #: Non parametric spline dimensions is currently not used.
    # spline_dimension: conint(ge=1)
    #: control points of the spline.
    control_points: Optional[List[List[VariableLocation]]]

    @root_validator(pre=True)
    @classmethod
    def make_default_control_point_grid(
        cls, values: Dict[str, Any]
    ) -> List[List[VariableLocation]]:
        """Validator for control_point_variables.

        If value is None a equidistant grid of control points will be given
        with variability of half the space between the each control point.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            ParserException: Emitted parser error.

        Returns:
            List[List[VariableLocation]]: Definition of the control_points
        """
        if not values.get("control_points"):
            if "space_dimensions" not in values.keys():
                raise ParserException(
                    "SplineDefinition",
                    "control_point_variables",
                    "During validation the prerequisite variable "
                    f"space_dimensions was not present. {str(values)}",
                )
            spline_dimensions = values["space_dimensions"]
            n_points_in_dim = [
                dim["number_of_points"] for dim in spline_dimensions
            ]
            n_points_in_dim.reverse()
            # this can be done in a smaller and faster footprint but for
            # readability and understanding
            dimension_lists = []
            dim_spacings = []
            # create value range in each dimension separately
            for n_p in n_points_in_dim:
                dimension_lists.append(list(np.linspace(0.0, 1.0, n_p)))
                dim_spacings.append(1.0 / ((n_p - 1) * 2))
            # create each control point for by concatenating each value in each
            # list with each value of all other lists
            save_location = str(values["save_location"])
            v = None
            for inner_dim, dim_spacing in zip(dimension_lists, dim_spacings):
                # first iteration the v vector must be initialized with the
                # first dimension vector
                if v is None:
                    v = []
                    for element in inner_dim:
                        if element == 0.0:
                            v.append(
                                VariableLocation(
                                    current_position=element,
                                    min_value=element,
                                    max_value=element + dim_spacing,
                                    save_location=save_location,
                                )
                            )
                        elif element == 1.0:
                            v.append(
                                VariableLocation(
                                    current_position=element,
                                    min_value=element - dim_spacing,
                                    max_value=element,
                                    save_location=save_location,
                                )
                            )
                        else:
                            v.append(
                                VariableLocation(
                                    current_position=element,
                                    min_value=element - dim_spacing,
                                    max_value=element + dim_spacing,
                                    save_location=save_location,
                                )
                            )
                # for each successive dimension for each existing element in v
                # each value in the new dimension must be added
                else:
                    temp_v = []
                    for current_list in v:
                        for element in inner_dim:
                            elem = copy.deepcopy(
                                current_list
                                if isinstance(current_list, list)
                                else [current_list]
                            )
                            if element == 0.0:
                                temp_v.append(
                                    [
                                        VariableLocation(
                                            current_position=element,
                                            min_value=element,
                                            max_value=element + dim_spacing,
                                            save_location=save_location,
                                        )
                                    ]
                                    + elem
                                )
                            elif element == 1.0:
                                temp_v.append(
                                    [
                                        VariableLocation(
                                            current_position=element,
                                            min_value=element - dim_spacing,
                                            max_value=element,
                                            save_location=save_location,
                                        )
                                    ]
                                    + elem
                                )
                            else:
                                temp_v.append(
                                    [
                                        VariableLocation(
                                            current_position=element,
                                            min_value=element - dim_spacing,
                                            max_value=element + dim_spacing,
                                            save_location=save_location,
                                        )
                                    ]
                                    + elem
                                )
                    v = temp_v
            if isinstance(v[0], list):
                values["control_points"] = v
            else:
                values["control_points"] = [[v_i] for v_i in v]
        return values

    def get_number_of_points(self) -> int:
        """Returns the number of points in the Spline.

        Currently the number of points in the spline is calculated by
        multiplying the number of points in each spline dimension.

        Returns:
            int: number of points in the spline
        """
        return np.prod([
            dimension.number_of_points for dimension in self.space_dimensions
        ])


class BSplineDefinition(SplineDefinition):
    """Definition of the BSpline implementation of the ReLeSO Toolbox."""

    def get_shape(self) -> BSpline:
        """Creates the BSpline from the definition given by the json file.

        Returns:
            BSpline: given by the #degrees and knot_vector in each
            space_dimension and the current control points.
        """
        self.get_logger().debug("Creating BSpline.")
        self.get_logger().debug(
            f"With control_points: {self.get_parameter_values()}"
        )

        return BSpline(
            [space_dim.degree for space_dim in self.space_dimensions],
            [
                space_dim.get_knot_vector()
                for space_dim in self.space_dimensions
            ],
            self.get_parameter_values(),
        )


class NURBSDefinition(SplineDefinition):
    """Definition of a NURBS spline.

    Definition of the NURBS implementation of the ReLeSO Toolbox, in
    comparison to the B-Spline implementation only an additional weights
    vector is added.
    """

    #: weights for the NURBS Spline definition. Other parameters are part of
    #: the spline definition class. Can be fixed or changeable
    weights: List[Union[float, VariableLocation]]

    @validator("weights")
    @classmethod
    def validate_weights(
        cls,
        v: Optional[List[Union[float, VariableLocation]]],
        values: Dict[str, Any],
    ) -> List[Union[float, VariableLocation]]:
        """Validator for variable weights.

        Validate if the correct number of weights are present in the weight
        vector. If weight vector is not given weight vector should be all ones.

        Args:
            v (List[Union[float, VariableLocation]]): Value to validate
            values (Dict[str, Any]): Previously validated variables

        Raises:
            ParserException: Parser Error

        Returns:
            List[Union[float, VariableLocation]]: Filled weight vector.
        """
        n_cp = np.prod([
            space_dim.number_of_points
            for space_dim in values["space_dimensions"]
        ])
        if isinstance(v, list):
            if len(v) == n_cp:
                get_parser_logger().debug(
                    "Found correct number of weights in SplineDefinition."
                )
            else:
                raise ParserException(
                    "SplineDefinition NURBS",
                    "weights",
                    f"The length of the "
                    f"weight vector {len(v)} is not the same as the number of "
                    f"control_points {n_cp}.",
                )

        # actually not functional since this would create a race condition
        # between BSpline and NURBS, since the only difference is the weights
        # variable.
        # elif v is None:
        #     v = list(np.ones(n_cp))

        return v

    @validator("weights", each_item=True)
    @classmethod
    def convert_weights_into_variable_location(
        cls, v: Union[float, VariableLocation], values: Dict[str, Any]
    ) -> VariableLocation:
        """Validator for variable weights.

        Convert all float values in the weight vector into VariableLocations.
        So that these can also be used as actions.

        Args:
            v (List[Union[float, VariableLocation]]): Value to validate
            values (Dict[str, Any]): Previously validated variables

        Returns:
            List[VariableLocation]: Filled weight vector.
        """
        if isinstance(v, float):
            save_location = str(values["save_location"])
            return VariableLocation(
                current_position=v, save_location=save_location
            )
        elif type(v) is VariableLocation:
            return v
        # validated by pydantic
        # else:
        #     raise ParserException(
        #         "SplineDefinition NURBS",
        #         "weights",
        #         "The weight vector contains a value which is not a float "
        #         f"or a VariableLocation. The type is {type(v)}.",
        #     )

    def get_weights(self) -> List[float]:
        """Returns the weights of the NURBS spline.

        Returns:
            List[float]: List of weights.
        """
        return [weight.current_position for weight in self.weights]

    def get_shape(self) -> NURBS:
        """Creates a NURBSSpline from the definition given by the json file.

        Returns:
            NURBS: NURBSSpline
        """
        self.get_logger().debug("Creating NURBS.")

        return NURBS(
            [space_dim.degree for space_dim in self.space_dimensions],
            [
                space_dim.get_knot_vector()
                for space_dim in self.space_dimensions
            ],
            self.get_parameter_values()[:-1],
            self.get_weights(),
        )

    def get_actions(self) -> List[VariableLocation]:
        """Extends the control point actions with the weight actions.

        Returns:
            List[VariableLocation]: List of possible actions for this NURBS
            spline.
        """
        actions = super().get_actions()
        actions.extend([
            variable for variable in self.weights if variable.is_action()
        ])
        # raise RuntimeError(f"Actions: {actions}")
        return actions

    def get_parameter_values(self) -> List[List[float]]:
        """Returns the current positions of all control points with weights as
        well.

        Returns:
            List[List[float]]: Positions of all control points.
        """
        control_points = super().get_parameter_values()
        control_points.append([
            weight.current_position for weight in self.weights
        ])
        return control_points

    def reset(self) -> None:
        """Resets the spline to the original shape."""
        super().reset()
        for weight in self.weights:
            weight.reset()
