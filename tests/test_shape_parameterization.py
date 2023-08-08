import numpy as np
import pydantic
import pytest

from releso.shape_parameterization import ShapeDefinition, VariableLocation
from releso.util.logger import VerbosityLevel, get_parser_logger

base_dict = {
    "save_location": "test",
}


@pytest.mark.parametrize(
    "current_position, error",
    [
        (0, False),
        (100, False),
        (-100, False),
        (0.001, False),
        (-0.001, False),
        (None, True),  # Need at least current_position to instantiate
    ],
)
def test_variable_location_instantiates_only_current_position(
    current_position, error
):
    """Test that VariableLocation instantiates"""
    # First check the error state.
    if error:
        with pytest.raises(pydantic.ValidationError):
            variable_location = VariableLocation(
                **{"current_position": current_position, **base_dict}
            )
        return
    # check default initiation
    variable_location = VariableLocation(
        **{"current_position": current_position, **base_dict}
    )
    assert variable_location is not None
    assert variable_location.current_position == current_position
    assert variable_location.is_action() is False
    assert variable_location.min_value == current_position
    assert variable_location.max_value == current_position
    assert variable_location.n_steps == 10
    assert current_position == variable_location._original_position


@pytest.mark.parametrize(
    "current_position, min_value, max_value, error",
    [
        (0, None, None, False),
        (0, -10, None, False),
        (-10, -12, -8, False),
        (-10, -8, None, True),
        (-10, -8, -7, True),
        (-10, None, -12, True),
    ],
)
def test_variable_location_min_max_location(
    current_position, min_value, max_value, error
):
    """Test for correct setting of the min and max values."""
    # check if error is emitted
    if error:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            variable_location = VariableLocation(
                **{
                    "current_position": current_position,
                    "min_value": min_value,
                    "max_value": max_value,
                    **base_dict,
                }
            )
        assert "or equal to the" in str(exc_info.value)
        return
    variable_location = VariableLocation(
        **{
            "current_position": current_position,
            "min_value": min_value,
            "max_value": max_value,
            **base_dict,
        }
    )
    if min_value is None:
        assert current_position == variable_location.min_value
    else:
        assert min_value == variable_location.min_value
    if max_value is None:
        assert current_position == variable_location.max_value
    else:
        assert max_value == variable_location.max_value


@pytest.mark.parametrize(
    "current_position, min_value, max_value, is_action",
    [
        (0, -1, 1, True),
        (0, 0, 1, True),
        (0, None, None, False),
        (0, -1, 0, True),
    ],
)
def test_variable_location_is_action(
    current_position, min_value, max_value, is_action
):
    variable_location = VariableLocation(
        **{
            "current_position": current_position,
            "min_value": min_value,
            "max_value": max_value,
            **base_dict,
        }
    )
    assert variable_location.is_action() is is_action


@pytest.mark.parametrize(
    "current_position, min_value, max_value, n_steps, step, expected_step, "
    "error",
    [
        (0, -1, 1, None, None, 0.2, False),
        (0, -1, 1, None, 0.1, 0.1, False),
        (0, -2, 2, 100, None, 4 / 100, False),
        (0, -2, 2, 100, 0.25, 0.25, False),
    ],
)
def test_variable_location_step_length(
    current_position, min_value, max_value, n_steps, step, expected_step, error
):
    """Test for correct setting of the n_steps and step values."""
    # check if error is emitted
    if error:
        with pytest.raises(pydantic.ValidationError) as exc_info:
            variable_location = VariableLocation(
                **{
                    "current_position": current_position,
                    "min_value": min_value,
                    "max_value": max_value,
                    "n_steps": n_steps,
                    "step": step,
                    **base_dict,
                }
            )
        assert "or equal to the" in str(exc_info.value)
        return
    variable_location = VariableLocation(
        **{
            "current_position": current_position,
            "min_value": min_value,
            "max_value": max_value,
            "n_steps": n_steps,
            "step": step,
            **base_dict,
        }
    )
    assert np.isclose(expected_step, variable_location.step)
    if n_steps is None and step is None:
        assert n_steps == variable_location.n_steps


@pytest.mark.parametrize(
    "current_position, min_value, max_value, step, warning",
    [
        (0, -4, 1, 0.2, False),
        (0, -1, 1, 4, True),  # step length is greater than the interval
        (0, -2, 2, 4, False),
        (0, -2, 2, 4.0001, True),  # step length is greater than the interval
    ],
)
def test_variable_location_step_length_warning(
    current_position, min_value, max_value, step, warning, caplog
):
    if warning:
        with caplog.at_level(VerbosityLevel.WARNING, logger="ReLeSO_parser"):
            variable_location = VariableLocation(
                **{
                    "current_position": current_position,
                    "min_value": min_value,
                    "max_value": max_value,
                    "step": step,
                    **base_dict,
                }
            )
            assert "is greater than the interval for" in caplog.text
    else:
        variable_location = VariableLocation(
            **{
                "current_position": current_position,
                "min_value": min_value,
                "max_value": max_value,
                "step": step,
                **base_dict,
            }
        )


@pytest.mark.parametrize(
    "current_position, min_value, max_value, n_steps, step, iterations, "
    "expect_position",
    [
        (
            0,
            None,
            None,
            None,
            None,
            [True] * 5,
            0,
        ),  # stay in bound upper limit
        (
            0,
            None,
            None,
            None,
            None,
            [False] * 5,
            0,
        ),  # stay in bound lower limit
        (0, -10, None, None, None, [True] * 5, 0),  # stay in bound upper limit
        (
            0,
            -10,
            None,
            None,
            None,
            [False] * 5,
            -5,
        ),  # correct value after iteration low
        (
            0,
            None,
            10,
            None,
            None,
            [True] * 5,
            5,
        ),  # correct value after iteration up
        (0, None, 10, None, None, [False] * 5, 0),  # lower bound w boundary
        (0, -10, None, None, None, [True] * 15, 0),  # upper bound w boundary
        (
            0,
            -10,
            None,
            None,
            None,
            [False] * 15,
            -10,
        ),  # lower bound w boundary
        (0, None, 10, None, None, [True] * 15, 10),  # upper bound w boundary
        (0, None, 10, None, None, [False] * 15, 0),  # lower bound w boundary
        (0, -1, 4, None, None, [True] * 3, 1.5),  # complex case
        (
            0,
            -1,
            4,
            None,
            None,
            [False, False, False, True, True, True, True, True],
            1.5,
        ),  # complex case back and forth
        (
            0,
            -1,
            4,
            None,
            None,
            [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
            ],
            3.5,
        ),  # complex case back and forth
    ],
)
def test_variable_location_discrete_action(
    current_position,
    min_value,
    max_value,
    n_steps,
    step,
    iterations,
    expect_position,
):
    variable_location = VariableLocation(
        **{
            "current_position": current_position,
            "min_value": min_value,
            "max_value": max_value,
            "n_steps": n_steps,
            "step": step,
            **base_dict,
        }
    )
    last_current_position = current_position
    for iteration in iterations:
        last_current_position = variable_location.apply_discrete_action(
            iteration
        )
    assert np.isclose(
        last_current_position, variable_location.current_position
    )
    assert np.isclose(expect_position, variable_location.current_position)
    # check if original position is correctly preserved
    assert current_position == variable_location._original_position
    # check if reset works as expected
    variable_location.reset()
    assert variable_location._original_position == current_position
    assert variable_location.current_position == current_position


@pytest.mark.parametrize(
    "current_position, min_value, max_value, n_steps, step, iterations, "
    "expect_position",
    [
        # iterations are values in range [-1,1] defining the value_range of
        # the variable_location
        (10, 5, 15, None, None, [1], 15),  # upper limit
        (10, 5, 15, None, None, [0], 10),  # middle, start from middle
        (10, 5, 15, None, None, [-1], 5),  # lower limit
        (0, None, None, None, None, [1.1, -1.1], 0),  # stay in bound
        (0, -10, None, None, None, [0, 2], 0),  # stay in bound upper
        (
            0,
            -10,
            None,
            None,
            None,
            [2, 0],
            -5,
        ),  # two step correct value exceed upper
        (
            0,
            None,
            10,
            None,
            None,
            [-2, 0],
            5,
        ),  # two step correct value exceed lower
        (0, None, 10, None, None, [0.8, -1.4], 0),  # two step lower limit
        (0, -10, None, None, None, [-1.8, 1], 0),  # two step upper limit
        (0, -1, 4, None, None, [-0.33, -0.467999999], 0.33),  # complex value
    ],
)
def test_variable_location_continuous_action(
    current_position,
    min_value,
    max_value,
    n_steps,
    step,
    iterations,
    expect_position,
):
    variable_location = VariableLocation(
        **{
            "current_position": current_position,
            "min_value": min_value,
            "max_value": max_value,
            "n_steps": n_steps,
            "step": step,
            **base_dict,
        }
    )
    last_current_position = current_position
    for iteration in iterations:
        last_current_position = variable_location.apply_continuous_action(
            iteration
        )
    assert np.isclose(
        last_current_position, variable_location.current_position
    )
    assert np.isclose(expect_position, variable_location.current_position)
    # check if original position is correctly preserved
    assert current_position == variable_location._original_position
    # check if reset works as expected
    variable_location.reset()
    assert variable_location._original_position == current_position
    assert variable_location.current_position == current_position


# test ShapeDefinition


@pytest.mark.parametrize(
    "control_points, n_elements, error",
    [
        ([[0.0], [1.0], [1.0]], 3, False),
        (
            [[0.0, 0.1, 0.2, 0.3, 0.4], [1.0], [1.0]],
            15,
            False,
        ),  # this might change
        ([[1.0]], 1, False),
        ([[(1.0, 2.0)]], 1, True),
        (
            [[VariableLocation(current_position=1, save_location="test")]],
            1,
            False,
        ),
    ],
)
def test_shapeDefinition_initiation(control_points, n_elements, error):
    if error:
        with pytest.raises(pydantic.ValidationError) as err:
            shape = ShapeDefinition(
                **{
                    "control_points": control_points,
                    "save_location": base_dict["save_location"],
                }
            )
            assert "either a float or a VariableLocation" in str(err.value)
        return
    shape = ShapeDefinition(
        **{
            "control_points": control_points,
            "save_location": base_dict["save_location"],
        }
    )
    all_variables_variableLocation = True
    for element in shape.control_points:
        for item in element:
            if not isinstance(item, VariableLocation):
                all_variables_variableLocation = False
                break
        if not all_variables_variableLocation:
            break
    assert all_variables_variableLocation

    assert shape.get_number_of_points() == n_elements

    cps = shape.get_control_points()
    all_the_same = True
    for original_cp_list, cp_list in zip(control_points, cps):
        for point_o, point in zip(original_cp_list, cp_list):
            if not (
                (
                    isinstance(point_o, VariableLocation)
                    and point_o.current_position == point
                )
                or point_o == point
            ):
                all_the_same = False
                break
        if not all_the_same:
            break
    assert all_the_same

    assert shape.get_shape() == shape.get_control_points()


@pytest.mark.parametrize(
    "control_points, n_actions",
    [
        (
            [
                [
                    VariableLocation(
                        current_position=0.1, min_value=0, save_location="test"
                    ),
                    VariableLocation(
                        current_position=0.2, min_value=0, save_location="test"
                    ),
                    VariableLocation(
                        current_position=0.3, min_value=0, save_location="test"
                    ),
                    VariableLocation(
                        current_position=0.4, min_value=0, save_location="test"
                    ),
                    VariableLocation(
                        current_position=0.5,
                        min_value=-1,
                        save_location="test",
                    ),
                ]
            ],
            5,
        ),
        (
            [
                [
                    VariableLocation(
                        current_position=0.1, save_location="test"
                    ),
                    VariableLocation(
                        current_position=0.2, save_location="test"
                    ),
                    VariableLocation(
                        current_position=0.3, min_value=0, save_location="test"
                    ),
                    VariableLocation(
                        current_position=0.4, min_value=0, save_location="test"
                    ),
                    VariableLocation(
                        current_position=0.5,
                        min_value=-1,
                        save_location="test",
                    ),
                ]
            ],
            3,
        ),
    ],
)
def test_shapeDefinition_get_actions(control_points, n_actions):
    shape = ShapeDefinition(
        **{
            "control_points": control_points,
            "save_location": base_dict["save_location"],
        }
    )
    actions = shape.get_actions()
    assert len(actions) == n_actions

    original_cps = shape.get_control_points()

    for action in actions:
        action.apply_continuous_action(0.2)

    for o_cp, cp in zip(original_cps, shape.get_control_points()):
        assert o_cp != cp

    shape.reset()

    for o_cp, cp in zip(original_cps, shape.get_control_points()):
        assert o_cp == cp
