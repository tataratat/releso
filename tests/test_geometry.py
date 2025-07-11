import copy

import numpy as np
import pytest
from gymnasium import spaces
from pydantic import ValidationError

from releso.geometry import FFDGeometry, Geometry
from releso.shape_parameterization import ShapeDefinition
from releso.spline import BSplineDefinition, NURBSDefinition
from releso.util.logger import VerbosityLevel


@pytest.mark.parametrize(
    (
        "shape_definition",
        "action_based",
        "discrete_action",
        "reset_random",
        "correct_type",
        "n_action_variables",
    ),
    [
        (
            "default_shape",
            None,
            None,
            None,
            ShapeDefinition,
            2,
        ),  # test that default is correct
        ("default_shape", True, True, True, ShapeDefinition, 2),
        ("default_shape", False, False, False, ShapeDefinition, 2),
        ("bspline_shape", None, False, None, BSplineDefinition, 18),
        ("nurbs_shape", None, False, None, NURBSDefinition, 20),
    ],
)
def test_geometry_init(
    shape_definition,
    action_based,
    discrete_action,
    reset_random,
    correct_type,
    n_action_variables,
    dir_save_location,
    caplog,
    request,
):
    call_dict = {
        "shape_definition": request.getfixturevalue(shape_definition),
        "save_location": dir_save_location,
    }
    if action_based is not None:
        call_dict["action_based_observation"] = action_based
    if discrete_action is not None:
        call_dict["discrete_actions"] = discrete_action
    if reset_random is not None:
        call_dict["reset_with_random_action_values"] = reset_random
    geometry = Geometry(**call_dict)
    assert isinstance(geometry.shape_definition, correct_type)
    assert geometry.action_based_observation == (
        action_based if action_based is not None else True
    )
    assert geometry.discrete_actions == (
        discrete_action if discrete_action is not None else True
    )
    assert geometry.reset_with_random_action_values == (
        reset_random if reset_random is not None else False
    )

    # check get all actions
    geometry.setup("id")

    # control points
    assert (
        geometry.get_parameter_values()
        == geometry.shape_definition.get_parameter_values()
    )

    assert geometry.is_geometry_changed() is False
    assert geometry.apply() == geometry.get_parameter_values()

    original_cps = geometry.get_parameter_values()

    # actions
    assert len(geometry._actions) == len(
        geometry.shape_definition.get_actions()
    )
    assert len(geometry._actions) == n_action_variables

    act_def = geometry.get_action_definition()
    if geometry.discrete_actions:
        assert isinstance(act_def, spaces.Discrete)
        assert act_def.n == 2 * len(geometry._actions)
        geometry.apply_action(3)
    else:
        assert isinstance(act_def, spaces.Box)
        assert act_def.shape == (len(geometry._actions),)
        geometry.apply_action(np.random.rand(len(geometry._actions)))
    assert geometry.is_geometry_changed() is True

    # observation
    if geometry.action_based_observation:
        assert geometry.get_observation_definition()[1].shape == (
            len(geometry._actions),
        )
        assert len(geometry.get_observation()) == len(geometry._actions)
    else:
        with caplog.at_level(VerbosityLevel.WARNING):
            geometry.get_observation_definition()
            assert (
                "Observation space is accessed which should not happen."
                in caplog.text
            )
            assert geometry.get_observation() is None

    # reset
    current_cps = geometry.get_parameter_values()
    assert current_cps != original_cps
    if geometry.reset_with_random_action_values:
        reset_cps = geometry.reset()
        assert reset_cps != current_cps
        assert reset_cps != original_cps
    else:
        assert geometry.reset() == original_cps


@pytest.mark.parametrize(
    ("discrete_actions"),
    [
        (True),
        (False),
    ],
)
def test_geometry_apply_random_action(
    dir_save_location,
    discrete_actions,
    default_shape_discrete_possible_values,
    request,
):
    call_dict = {
        "shape_definition": request.getfixturevalue("default_shape"),
        "save_location": dir_save_location,
        "discrete_actions": discrete_actions,
    }
    geometry = Geometry(**call_dict)
    geometry.setup("id")
    # random action with seed produces same result
    geometry.apply_random_action("asd")
    set_1 = copy.deepcopy([
        action.current_position for action in geometry._actions
    ])

    geometry.apply_random_action("asd")
    set_2 = copy.deepcopy([
        action.current_position for action in geometry._actions
    ])
    if discrete_actions:
        for i, action in enumerate(geometry._actions):
            assert (
                action.current_position
                in (default_shape_discrete_possible_values[i])
            ), (
                f"Action {i} has value {action.current_position} which is not in "
                f"the possible values {default_shape_discrete_possible_values[i]}."
            )
    assert np.allclose(set_1, set_2)


@pytest.mark.parametrize(
    (
        "shape_definition",
        "action_based",
        "discrete_action",
        "reset_random",
        "load_sample_file",
        "export_mesh",
        "error",
    ),
    [
        (
            "default_shape",
            False,
            False,
            False,
            "volumes/tet/3DBrickTet.msh",
            None,
            "FFD can only be performed with a splinepy Spline",
        ),
        (
            "bspline_shape",
            None,
            False,
            None,
            "faces/quad/2DChannelQuad.msh",
            {"format": "mixd", "export_path": "test"},
            False,
        ),
        (
            "nurbs_shape",
            None,
            False,
            None,
            "faces/quad/2DChannelQuad.msh",
            None,
            False,
        ),
    ],
    indirect=["load_sample_file"],
)
def test_ffd_geometry_init(
    shape_definition,
    action_based,
    discrete_action,
    reset_random,
    load_sample_file,
    export_mesh,
    error,
    dir_save_location,
    caplog,
    request,
):
    call_dict = {
        "shape_definition": request.getfixturevalue(shape_definition),
        "save_location": dir_save_location,
        "mesh": {
            "path": load_sample_file,
            "save_location": dir_save_location,
            "dimensions": 2,
        },
    }
    if export_mesh is not None:
        export_mesh["save_location"] = dir_save_location
        call_dict["export_mesh"] = export_mesh
    if action_based is not None:
        call_dict["action_based_observation"] = action_based
    if discrete_action is not None:
        call_dict["discrete_actions"] = discrete_action
    if reset_random is not None:
        call_dict["reset_with_random_action_values"] = reset_random

    if error:
        with pytest.raises(ValidationError) as err:
            geometry = Geometry(**call_dict)
            assert error in str(err.value)
        return

    geometry = FFDGeometry(**call_dict)
    assert geometry.action_based_observation == (
        action_based if action_based is not None else True
    )
    assert geometry.discrete_actions == (
        discrete_action if discrete_action is not None else True
    )
    assert geometry.reset_with_random_action_values == (
        reset_random if reset_random is not None else False
    )

    with caplog.at_level(VerbosityLevel.INFO):
        # This warning should happen
        geometry.setup("id")
        assert "Found empty dimension" in caplog.text
    # control points
    assert (
        geometry.get_parameter_values()
        == geometry.shape_definition.get_parameter_values()
    )

    assert geometry.is_geometry_changed() is False
    geometry.apply()

    _ = geometry.get_parameter_values()

    # actions
    assert len(geometry._actions) == len(
        geometry.shape_definition.get_actions()
    )
    act_def = geometry.get_action_definition()
    if geometry.discrete_actions:
        assert isinstance(act_def, spaces.Discrete)
        assert act_def.n == 2 * len(geometry._actions)
        geometry.apply_action(3)
    else:
        assert isinstance(act_def, spaces.Box)
        assert act_def.shape == (len(geometry._actions),)
        geometry.apply_action(np.random.rand(len(geometry._actions)))
    assert geometry.is_geometry_changed()


@pytest.mark.parametrize(
    (
        "shape_definition",
        "correct_type",
        "n_action_variables",
        "observation_shape",
    ),
    [
        (
            "default_shape",
            ShapeDefinition,
            2,
            [5, 5],
        ),  # test that default is correct
        (
            ["default_shape", "default_shape"],
            [ShapeDefinition, ShapeDefinition],
            4,
            [[5, 5], [5, 5]],
        ),
        (
            ["default_shape", "bspline_shape"],
            [ShapeDefinition, BSplineDefinition],
            20,
            [[5, 5], [2, 2, 2, 2, 2, 2, 2, 2, 2]],
        ),
        (
            ["default_shape", "nurbs_shape"],
            [ShapeDefinition, NURBSDefinition],
            22,
            [[5, 5], [2, 2, 2, 2, 2, 2, 2, 2, 2, 9]],
        ),
        # if nurbs and b splines are mixed the shape is always a nurbs.
        (
            ["bspline_shape", "nurbs_shape", "default_shape"],
            [BSplineDefinition, NURBSDefinition, ShapeDefinition],
            40,
            [
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
                [5, 5],
            ],
        ),
    ],
)
def test_geometry_shape_list(
    shape_definition,
    correct_type,
    n_action_variables,
    observation_shape,
    dir_save_location,
    request,
):
    call_dict = {
        "save_location": dir_save_location,
    }
    if isinstance(shape_definition, list):
        call_dict["shape_definition"] = [
            request.getfixturevalue(shape) for shape in shape_definition
        ]
    else:
        call_dict["shape_definition"] = request.getfixturevalue(
            shape_definition
        )

    geometry = Geometry(**call_dict)
    # correct type of shape definition
    if isinstance(correct_type, list):
        assert isinstance(geometry.shape_definition, list), (
            "Shape definition should be a list."
        )
        assert len(geometry.shape_definition) == len(correct_type), (
            "Shape definition should have the correct length."
        )
        for shape, correct in zip(geometry.shape_definition, correct_type):
            assert isinstance(shape, correct), (
                f"Shape definition should be of type {correct}, but is {[type(s) for s in geometry.shape_definition]} {correct_type}."
            )
    else:
        assert isinstance(geometry.shape_definition, correct_type), (
            f"Shape definition should be of type {correct_type}, but is {type(geometry.shape_definition)}."
        )

    # correct action shape

    geometry.setup("id")

    assert len(geometry._actions) == n_action_variables

    # correct observation shape
    def check_shape(item, shape):
        assert len(item) == len(shape), (
            f"Length of item and b must be equal. Got {item} and {shape}."
        )
        for i, s in zip(item, shape):
            if isinstance(s, list):
                check_shape(i, s)

    param_values = geometry.get_parameter_values()

    check_shape(param_values, observation_shape)
