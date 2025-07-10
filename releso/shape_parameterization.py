"""Definition of the basic shape classes of releso.

In ReLeSO shapes mostly define what the actions in the RL problem are.
Due to that this framework is intended for shape optimization tasks the name
shape was chosen.

Note:
    To perform Free Form Deformation (FFD) splines are needed. These are given
    in a separate module, please see the file `spline.py` for these shape
    definitions.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic.class_validators import root_validator, validator
from pydantic.fields import PrivateAttr
from pydantic.types import conint

from releso.base_model import BaseModel
from releso.exceptions import ParserException
from releso.util.logger import get_parser_logger


class VariableLocation(BaseModel):
    """Variable location class.

    Object of this class defines the position and movement possibilities for a
    single dimensions of a single control point of the geometry.
    """

    #: coordinate of the current position of the current control point in the
    #: dimension
    current_position: float
    #: lower bound of possible values which can be reached by this variable
    min_value: Optional[float] = None
    #: upper bound  of possible values which can be reached by this variable
    max_value: Optional[float] = None
    #: number of steps the value range is divided into used to define the step
    #: if not given.
    n_steps: Optional[conint(ge=1)] = 10
    #: If discrete actions are used the step is used to define the new current
    #: position by adding/subtracting it from the current position.
    step: Optional[float] = None

    # non json variables
    #: Is true if min_value and max_value are not the same value
    _is_action: Optional[bool] = PrivateAttr(default=None)
    #: Original position needs to be saved so that the shape can be easily
    #: reset to its original state.
    _original_position: Optional[float] = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        """Constructor for VariableLocation."""
        super().__init__(**data)
        self._original_position = self.current_position

    @validator("min_value", "max_value", always=True)
    @classmethod
    def set_variable_to_current_position_if_not_given(
        cls, v, values, field
    ) -> float:
        """Validator for min_value and max_value.

        Validation of the min and max values for the current VariableLocation.
        If non are set no variability is assumed min = max = current_position

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Raises:
            ParserException: Parser error if current_position is not already
            validated.

        Returns:
            float: value of the validated value.
        """
        if v is None:
            if "current_position" in values.keys():
                return values["current_position"]
            else:
                raise ParserException(
                    "VariableLocation",
                    field,
                    "Please correctly define the current position.",
                )
        return v

    @validator("max_value", always=True)
    @classmethod
    def check_max_value_is_greater_than_current_value(
        cls, v, values, field
    ) -> float:
        """Validates that the max value is greater-equal to the current value.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Returns:
            float: value of the validated value.
        """
        if v < values["current_position"]:
            raise ParserException(
                "VariableLocation",
                field,
                f"The current_value {values['current_position']} must be"
                f" smaller or equal to the max_value {v}.",
            )
        return v

    @validator("min_value", always=True)
    @classmethod
    def check_min_value_is_smaller_than_current_value(
        cls, v, values, field
    ) -> float:
        """Validates that the min value is smaller-equal to the current value.

        Args:
            v ([type]): Value to validate.
            values ([type]): Already validated values.
            field ([type]): Name of the field that is currently validated.

        Returns:
            float: value of the validated value.
        """
        if v > values["current_position"]:
            raise ParserException(
                "VariableLocation",
                field,
                f"The current_value {values['current_position']} must be "
                f"bigger or equal to the max_value {v}.",
            )
        return v

    @root_validator
    @classmethod
    def define_step(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validator for class.

        Validation function that defines the step taken if the action would be
        discrete. If nothing is given for the calculation a n_steps default
        value of 10 is used.

        Args:
            values (Dict[str, Any]): Validated variables of the object.

        Returns:
            Dict[str, Any]: Validated variables but if steps was not given
            before now it has a value.
        """
        step, n_steps, min_value, max_value = (
            values.get("step"),
            values.get("n_steps"),
            values.get("min_value"),
            values.get("max_value"),
        )
        value_range = max_value - min_value
        if step is not None:
            if step > value_range:
                get_parser_logger().warning(
                    f"The defined step {step} is greater than the interval for"
                    f" this variable {value_range}. This is not intended "
                    "behavior."
                )
            return values  # if step length is defined ignore n_steps
        elif n_steps is None:
            # if discrete but neither step nor n_steps is defined a default 10
            # steps are assumed.
            n_steps = 10
        step = value_range / n_steps
        values["step"] = step
        values["n_steps"] = n_steps
        return values

    def is_action(self) -> bool:
        """Boolean if variable defines an action.

        Checks if the variable has variability. Meaning is there optimization
        between the min_value and the max_value.

        Returns:
            bool: True if (self.max_value > self.current_position) or
            (self.min_value < self.current_position)
        """
        if self._is_action is None:
            self._is_action = (self.max_value > self.current_position) or (
                self.min_value < self.current_position
            )
        return self._is_action

    def apply_discrete_action(self, increasing: bool) -> float:
        """Apply a discrete action to the variable.

        Apply the a discrete action to the current value. If resulting value
        would be outside of the min and max value range no step is taken.

        Args:
            increasing (bool): Toggle whether or not the value should increase
            or not.

        Returns:
            float: Value of the position that the current position is now at.
        """
        step = self.step if increasing else -self.step
        if not (
            self.min_value <= self.current_position + step <= self.max_value
        ):
            step = 0.0
        self.current_position += step
        return self.current_position

    def apply_continuous_action(self, value: float) -> float:
        """Apply a continuous action the variable.

        Apply the zero-mean normalized value as the new current position.
        Needs to descale the value first. Also applies clipping.

        Args:
            value (float): Scaled value [-1,1] that needs to be descaled and
            applied as the new current position.

        Returns:
            float: New clipped value.
        """
        delta = self.max_value - self.min_value
        descaled_value = ((value + 1.0) / 2.0) * delta
        self.current_position = np.clip(
            descaled_value + self.min_value, self.min_value, self.max_value
        )
        return self.current_position

    def reset(self) -> None:
        """Resets the current_location to the initial position."""
        if self._original_position is not None:
            self.current_position = self._original_position


class ShapeDefinition(BaseModel):
    """Base of shape parameterization, also represents a simple point cloud."""

    #: control_points of the shape. These are the base variables used for the
    #: optimization. Overwrite `get_actions` and `get_parameter_values` if
    #: additional optimization variables are needed. See (WIP) NURBSDefinition.
    control_points: List[List[VariableLocation]]

    def get_number_of_points(self) -> int:
        """Returns the number of points in the Cube.

        Number of control points multiplied by the number of dimensions for
        each cp. Assumes that all dimensions have the same number of control
        points.

        Returns:
            int: number of points in the geometry
        """
        return int(len(self.control_points) * len(self.control_points[0]))

    @validator("control_points", each_item=True, pre=True)
    @classmethod
    def convert_all_control_point_locations_to_variable_locations(
        cls, v, values
    ):
        """Validator control_points.

        Converts all control points values into VariableLocations if the value
        is given as a simple float. Simple float will be converted into
        VariableLocation with current_position=value and no variability.

        Args:
            v ([type]): value to validate
            values ([type]): already validated values

        Returns:
            [type]: validated value
        """
        new_list = []
        for element in v:
            if type(element) is VariableLocation:
                new_list.append(element)
            elif type(element) in [dict, OrderedDict]:
                new_list.append(VariableLocation(**element))
            else:
                try:
                    element = float(element)
                    new_list.append(
                        VariableLocation(
                            current_position=element,
                            save_location=values["save_location"],
                        )
                    )
                except (ValueError, TypeError):
                    raise ParserException(
                        "ShapeDefinition",
                        "control_points",
                        "The control_points need to be either a float castable"
                        f" or a VariableLocation. Is of type {type(element)}.",
                    ) from None
        return new_list

    def get_parameter_values(self) -> List[List[float]]:
        """Returns the current positions of all control points.

        Returns:
            List[List[float]]: Positions of all control points.
        """
        return [
            [control_point.current_position for control_point in sub_list]
            for sub_list in self.control_points
        ]

    def get_actions(self) -> List[VariableLocation]:
        """Returns the action defined.

        Returns list of VariableLocations but only if the variable location is
        actually variable.

        Returns:
            List[VariableLocation]: See above
        """
        self.get_logger().debug(
            "Collecting all actions from shape parameterization."
        )
        return [
            variable
            for sub_dim in self.control_points
            for variable in sub_dim
            if variable.is_action()
        ]

    def get_shape(self) -> Any:
        """Generates the current shape.

        Here the control points are returned since there is no other meaning to
        this shape.

        Returns:
            Any: Shape that is generated.
        """
        return self.get_parameter_values()

    def reset(self) -> None:
        """Resets the shape to the original values."""
        for cp in self.control_points:
            for variable in cp:
                variable.reset()

    def draw_action_space(
        self,
        save_location: Optional[str] = None,
        no_axis: bool = False,
        fig_size: List[float] = None,
        dpi: int = 400,
    ):  # pragma: no cover
        """Draw the action space of the defined shape as a matplotlib figure.

        Needs to be reimplemented in subclasses where non control point
        variables are also used as actions. And if the control points are
        have more than 2D. The figure will plotted with matplotlib.

        Args:
            save_location (Optional[str], optional):
                Location to save the resulting figure to. Defaults to None.
            no_axis (bool, optional):
                Remove axis from the resulting figure. Defaults to False.
            fig_size (List[float], optional):
                Size of the resulting figure. Defaults to [6, 6].
            dpi (int, optional):
                DPI of the resulting figure. Defaults to 400.

        Raises:
            RuntimeError:
                Error is thrown if the control point's dimensionality is to
                high.
        """
        if fig_size is None:
            fig_size = [6, 6]
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        control_points = self.control_points
        if len(self.control_points[0]) > 2:
            raise RuntimeError(
                "Could not draw the shapes action space. Only a 2D parametric"
                " space is currently available."
            )
        elif len(self.control_points[0]) == 1:
            for idx, cp in enumerate(control_points):
                cp.insert(
                    0,
                    VariableLocation(
                        current_location=float(idx),
                        save_location=self.save_location,
                    ),
                )
        phi = np.linspace(0, 2 * np.pi, len(control_points))
        rgb_cycle = np.vstack((  # Three sinusoids
            0.5 * (1.0 + np.cos(phi)),  # scaled to [0,1]
            0.5 * (1.0 + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
            0.5 * (1.0 + np.cos(phi - 2 * np.pi / 3)),
        )).T  # Shape = (60,3)
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        dots = [[], []]

        for elem, color in zip(control_points, rgb_cycle):
            dots[0].append(elem[0].current_position)
            dots[1].append(elem[1].current_position)
            cur_pos = np.array([dots[0][-1], dots[1][-1]])
            spanning_elements = []
            no_boundary = False
            for item in elem:
                spanning_elements.append([item.min_value, item.max_value])
                if item.min_value == item.max_value:
                    no_boundary = True
            boundary = []
            for i, j in zip([0, 1, 1, 0], [0, 0, 1, 1]):
                end_pos = np.array([
                    spanning_elements[0][i],
                    spanning_elements[1][j],
                ])
                if not np.isclose(cur_pos, end_pos).all():  # draw arrow
                    difference = end_pos - cur_pos
                    ax.arrow(
                        cur_pos[0],
                        cur_pos[1],
                        difference[0] * 0.9,
                        difference[1] * 0.9,
                        width=0.005,
                        color=color,
                    )
                boundary.append(end_pos)
            if not no_boundary:
                pol = Polygon(
                    boundary, facecolor=color, linewidth=1, alpha=0.2
                )
                ax.add_patch(pol)
        ax.scatter(dots[0], dots[1], c=rgb_cycle, marker="o", s=50, zorder=3)
        if no_axis:
            plt.axis("off")
        if save_location:
            fig.savefig(save_location, transparent=True)
            plt.close()
