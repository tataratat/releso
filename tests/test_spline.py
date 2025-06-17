import copy

import numpy as np
import pytest
from pydantic import ValidationError

from releso.shape_parameterization import VariableLocation
from releso.spline import (
    BSplineDefinition,
    NURBSDefinition,
    SplineDefinition,
    SplineSpaceDimension,
)
from releso.util.logger import VerbosityLevel


@pytest.mark.parametrize(
    (
        "n_points",
        "degree",
        "knot_vector",
        "expected_knot_vector",
        "error",
        "msg",
        "verbosity_level",
    ),
    [
        (
            2,
            1,
            None,
            [0, 0, 1, 1],
            False,
            "The knot_vector for dimension test is not given",
            VerbosityLevel.DEBUG,
        ),
        (
            2,
            1,
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            False,
            "The knot_vector for dimension test is given.",
            VerbosityLevel.DEBUG,
        ),
        (
            2,
            None,
            None,
            [0, 0, 1, 1],
            True,
            "During validation the prerequisite",
            None,
        ),
        (
            3,
            1,
            None,
            [0, 0, 0.5, 1, 1],
            False,
            None,
            None,
        ),
        (
            2,
            1,
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            True,
            "The knot vector does not contain the correct number of items",
            VerbosityLevel.DEBUG,
        ),
        (
            3,
            4,
            None,
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            False,
            "Knot vector is created by adding the starting and ending parts.",
            VerbosityLevel.WARNING,
        ),
        (
            4,
            2,
            None,
            [0, 0, 0, 0.5, 1, 1, 1],
            False,
            None,
            None,
        ),
    ],
)
def test_spline_space_dimension_knot_vector(
    n_points,
    degree,
    knot_vector,
    expected_knot_vector,
    error,
    msg,
    verbosity_level,
    dir_save_location,
    caplog,
):
    calling_dict = {
        "save_location": dir_save_location,
        "name": "test",
        "number_of_points": n_points,
    }
    if degree:
        calling_dict["degree"] = degree
    if knot_vector:
        calling_dict["knot_vector"] = knot_vector

    if error:
        with pytest.raises(ValidationError) as error:
            spline_space_dimension = SplineSpaceDimension(**calling_dict)
        assert msg in str(error.value)
        return
    if msg:
        with caplog.at_level(verbosity_level, logger="ReLeSO_parser"):
            spline_space_dimension = SplineSpaceDimension(**calling_dict)
        assert msg in caplog.text
    else:
        spline_space_dimension = SplineSpaceDimension(**calling_dict)

    assert spline_space_dimension.knot_vector == expected_knot_vector
    assert spline_space_dimension.get_knot_vector() == expected_knot_vector


@pytest.mark.parametrize(
    "space_dimensions, control_points, wanted_control_points, n_points, error",
    [
        (None, None, None, None, True),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            [[21.0], [123.0], [124.0]],
            {
                "current_position": [[21], [123], [124]],
                "min_value": [[21], [123], [124]],
                "max_value": [[21], [123], [124]],
            },
            3,
            False,
        ),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            None,
            {
                "current_position": [[0.0], [0.5], [1.0]],
                "min_value": [[0.0], [0.25], [0.75]],
                "max_value": [[0.25], [0.75], [1.0]],
            },
            3,
            False,
        ),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
                {"name": "test", "number_of_points": 4, "degree": 1},
            ],
            None,
            {
                "current_position": [
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [1.0, 0.0],
                    [0.0, 1 / 3],
                    [0.5, 1 / 3],
                    [1.0, 1 / 3],
                    [0.0, 2 / 3],
                    [0.5, 2 / 3],
                    [1.0, 2 / 3],
                    [0.0, 1],
                    [0.5, 1],
                    [1.0, 1],
                ],
                "min_value": [
                    [0.0, 0.0],
                    [0.25, 0.0],
                    [0.750, 0.0],
                    [0.0, 1 / 6],
                    [0.25, 1 / 6],
                    [0.75, 1 / 6],
                    [0, 0.5],
                    [0.25, 0.5],
                    [0.750, 0.5],
                    [0.0, 5 / 6],
                    [0.25, 5 / 6],
                    [0.75, 5 / 6],
                ],
                "max_value": [
                    [0.25, 1 / 6],
                    [0.75, 1 / 6],
                    [1, 1 / 6],
                    [0.25, 0.5],
                    [0.75, 0.5],
                    [1, 0.5],
                    [0.25, 5 / 6],
                    [0.75, 5 / 6],
                    [1, 5 / 6],
                    [0.25, 1],
                    [0.75, 1],
                    [1, 1],
                ],
            },
            12,
            False,
        ),
    ],
)
def test_spline_definition_initiate(
    space_dimensions,
    control_points,
    wanted_control_points,
    n_points,
    error,
    dir_save_location,
):
    calling_dict = {
        "save_location": dir_save_location,
    }
    # only add these variables if they are not None
    if space_dimensions:
        calling_dict["space_dimensions"] = space_dimensions
    if control_points:
        calling_dict["control_points"] = control_points
    # check if error is expected
    if error:
        with pytest.raises(ValidationError) as error:
            SplineDefinition(**calling_dict)
        assert " the prerequisite variable space_dimensions" in str(
            error.value
        )
        return
    # initiate the spline definition
    spline_definition = SplineDefinition(**calling_dict)
    for o_var, o_w_cur, o_w_min, o_w_max in zip(
        spline_definition.control_points,
        *[wanted_control_points[key] for key in wanted_control_points.keys()],
    ):
        assert len(o_var) == len(o_w_cur) == len(o_w_min) == len(o_w_max)
        for var, w_cur, w_min, w_max in zip(
            o_var,
            o_w_cur,
            o_w_min,
            o_w_max,
        ):
            assert np.isclose(var.current_position, w_cur)
            assert np.isclose(var.min_value, w_min)
            assert np.isclose(var.max_value, w_max)
    assert spline_definition.get_number_of_points() == n_points


@pytest.mark.parametrize(
    "space_dimensions, control_points",
    [
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            None,
        ),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            [[21.0], [123.0], [124.0]],
        ),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
                {"name": "test", "number_of_points": 4, "degree": 1},
            ],
            None,
        ),
    ],
)
def test_bspline_definition_initiate(
    space_dimensions,
    control_points,
    dir_save_location,
):
    init_dict = {
        "save_location": dir_save_location,
    }
    if space_dimensions:
        init_dict["space_dimensions"] = space_dimensions
    if control_points:
        init_dict["control_points"] = control_points
    spline_def = SplineDefinition(**init_dict)
    bspline_def = BSplineDefinition(**init_dict)
    assert bspline_def.control_points == spline_def.control_points
    assert (
        bspline_def.get_number_of_points() == spline_def.get_number_of_points()
    )
    for bspline_dim, spline_dim in zip(
        bspline_def.space_dimensions, spline_def.space_dimensions
    ):
        assert bspline_dim.name == spline_dim.name
        assert bspline_dim.number_of_points == spline_dim.number_of_points
        assert bspline_dim.degree == spline_dim.degree
        assert bspline_dim.knot_vector == spline_dim.knot_vector

    bspline = bspline_def.get_shape()

    for b_kv, spline_dim in zip(
        bspline.knot_vectors, spline_def.space_dimensions
    ):
        assert np.allclose(b_kv, spline_dim.knot_vector)

    assert np.allclose(
        bspline.control_points, spline_def.get_parameter_values()
    )


@pytest.mark.parametrize(
    (
        "space_dimensions",
        "control_points",
        "weights",
        "wanted_weights",
        "error",
        "convert",
    ),
    [
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            None,
            None,
            None,
            "weights\n  field required",
            False,
        ),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            None,
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            False,
            False,
        ),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            None,
            [0.5, 0.5, 0.5, 0.4],
            None,
            "The length of the weight vector 4 is not the same as the number",
            False,
        ),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            None,
            [0.5, 0.5, 0.4],
            None,
            False,
            True,
        ),
        (
            [
                {"name": "test", "number_of_points": 3, "degree": 1},
            ],
            None,
            ["0.5", "asdsad", 0.4],
            None,
            "value is not a valid float",
            False,
        ),
        # (
        #     [
        #         {"name": "test", "number_of_points": 3, "degree": 1},
        #     ],
        #     [[21.0], [123.0], [124.0]],
        #     None,
        #     None,
        #     False,
        # ),
        # (
        #     [
        #         {"name": "test", "number_of_points": 3, "degree": 1},
        #         {"name": "test", "number_of_points": 4, "degree": 1},
        #     ],
        #     None,
        #     None,
        #     None,
        #     False,
        # ),
    ],
)
def test_nurbs_definition_initiate(
    space_dimensions,
    control_points,
    weights,
    wanted_weights,
    error,
    convert,
    dir_save_location,
):
    if convert:
        if weights:
            weights = [
                VariableLocation(
                    current_position=element,
                    min_value=element - 1,
                    max_value=element,
                    save_location=dir_save_location,
                )
                for element in weights
            ]
    init_dict = {
        "save_location": dir_save_location,
    }
    if space_dimensions:
        init_dict["space_dimensions"] = space_dimensions
    if control_points:
        init_dict["control_points"] = control_points
    spline_def = SplineDefinition(**copy.deepcopy(init_dict))
    if weights:
        init_dict["weights"] = weights
    if error:
        with pytest.raises(ValidationError) as err:
            NURBSDefinition(**init_dict)
        assert error in str(err.value)
        return
    nurbs_def = NURBSDefinition(**init_dict)
    assert nurbs_def.control_points == spline_def.control_points
    assert (
        nurbs_def.get_number_of_points() == spline_def.get_number_of_points()
    )
    for nurbs_dim, spline_dim in zip(
        nurbs_def.space_dimensions, spline_def.space_dimensions
    ):
        assert nurbs_dim.name == spline_dim.name
        assert nurbs_dim.number_of_points == spline_dim.number_of_points
        assert nurbs_dim.degree == spline_dim.degree
        assert nurbs_dim.knot_vector == spline_dim.knot_vector

    nurbs = nurbs_def.get_shape()

    for b_kv, spline_dim in zip(
        nurbs.knot_vectors, spline_def.space_dimensions
    ):
        assert np.allclose(b_kv, spline_dim.knot_vector)

    assert np.allclose(nurbs.control_points, spline_def.get_parameter_values())
    actions = nurbs_def.get_actions()
    assert len(actions) == (6 if convert else 3)
    original_actions = copy.deepcopy(actions)
    for action in actions:
        action.apply_discrete_action(False)
    assert not np.allclose(
        [action.current_position for action in actions],
        [action.current_position for action in original_actions],
    )
    nurbs_def.reset()
    assert np.allclose(
        [action.current_position for action in actions],
        [action.current_position for action in original_actions],
    )
