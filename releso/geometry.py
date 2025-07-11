"""Definition of the geometry to be optimized, which informs the action space.

File holds all classes which define the geometry and with that also the action
definition of the problem.
"""

from typing import Any, List, Optional, Tuple, Union
from uuid import UUID

import numpy as np
from gymnasium import spaces
from pydantic import PrivateAttr

from releso.base_model import BaseModel
from releso.exceptions import ParserException
from releso.mesh import MeshExporter, MeshTypes
from releso.shape_parameterization import ShapeDefinition, VariableLocation
from releso.spline import BSplineDefinition, NURBSDefinition, SplineDefinition
from releso.util.types import ObservationType

try:
    from splinepy.helpme.ffd import FFD
except ImportError as err:  # pragma: no cover
    from releso.util.module_import_raiser import ModuleImportRaiser

    FFD = ModuleImportRaiser("splinepy - FFD", err)

ShapeTypes = Union[ShapeDefinition, BSplineDefinition, NURBSDefinition]


class Geometry(BaseModel):
    """Definition of the Geometry."""

    #: Definition of the shape used for the geometry.
    #: This is the shape that will be used to define the geometry and
    #: the action space. If only a single shape is used this can be
    #: a single shape definition, if multiple shapes are used this can be
    #: a list of shape definitions.
    shape_definition: Union[ShapeTypes, List[ShapeTypes]]
    #: use the action space for the observations. If this is set to False
    #: please note that you need to define your own observation via a
    #: :py:class:`releso.spor.SPORObject`
    action_based_observation: bool = True
    #: Whether or not to use discrete actions if False continuous actions
    #: will be used.
    discrete_actions: bool = True
    #: Whether or not to reset the controllable control point variables to a
    #: random state (True) or the default original state (False). If
    #: discrete actions are used this will reset the action values to
    #: valid discrete values.
    reset_with_random_action_values: bool = False

    #: saved list of all available actions
    _actions: List[VariableLocation] = PrivateAttr()
    #: positions of all actions values of the previous step
    _last_actions: List[VariableLocation] = PrivateAttr(default=None)

    def _get_actions(self) -> List[VariableLocation]:
        """Get the actions defined by the shape definition.

        Returns:
            List[VariableLocation]: List of all actions defined by the
                shape_definition.
        """
        if isinstance(self.shape_definition, list):
            return [
                action
                for shape in self.shape_definition
                for action in shape.get_actions()
            ]
        return self.shape_definition.get_actions()

    def __init__(self, **data: Any) -> None:
        """Definition of the Geometry.

        Base geometry which just represents the used shape.

        Geometry also handles how the actions are defined and applied to the
        shape.
        """
        super().__init__(**data)
        self._actions = self._get_actions()

    def setup(self, environment_id: UUID):
        """Setup with additional information from the environment.

        Base geometry has no changes.

        Args:
            environment_id (UUID): Environment id.
        """
        self._actions = self._get_actions()

    def get_parameter_values(self) -> List[List[float]]:
        """Return all control_points of the spline.

        Returns:
            List[List[float]]: Nested list of control_points.
        """
        if isinstance(self.shape_definition, list):
            return [
                shape.get_parameter_values() for shape in self.shape_definition
            ]
        else:
            return self.shape_definition.get_parameter_values()

    def apply_action(self, action: Union[List[float], int]) -> Optional[Any]:
        """Function that applies a given action to the Spline.

        Args:
            action ([type]):  Action value depends on if the ActionSpace is
                              discrete (int - Signifier of the action) or
                              Continuous (List[float] - Value for each
                              continuous variable.)
        """
        self._last_actions = [act.current_position for act in self._actions]
        self.get_logger().debug(f"Applying action {action}")
        if self.discrete_actions:
            increasing: bool = action % 2 == 0
            action_index: int = int(action / 2)
            self.get_logger().debug(
                f"Setting discrete action, of variable {action_index}."
            )
            self._actions[action_index].apply_discrete_action(increasing)
            # TODO check if this is correct
        else:
            for new_value, action_obj in zip(action, self._actions):
                action_obj.apply_continuous_action(new_value)
        return self.apply()

    def apply(self) -> Optional[Any]:
        """Function which applies the current action values to the geometry.

        Overwrite if the geometry is represented by more than the shape
        representation.

        For the basic geometry no changes to the geometry are necessary. Since
        geometry is given by the shape directly.

        It also returns the basic information about the geometry. In this basic
        case it just returns the control_points of the shape. Do not rely on
        it returning the control_points since this might not be the case for
        subclasses. If you need the control_points use the relevant function.

        Returns:
            List[List[float]]: Current control points of the shape.
        """
        return self.get_parameter_values()

    def is_geometry_changed(self) -> bool:
        """Checks if the geometry was changed with the previous action apply.

        This function uses the actions as a metric on whether the geometry was
        changed. If the actions were not changed the function will return
        `False`, if the actions were changed it will return `True`.

        Needs to be reimplemented if other factors influence the geometry.

        Returns:
            bool: See above.
        """
        if self._last_actions is None:
            return False
        return not np.allclose(
            np.array([act.current_position for act in self._actions]),
            np.array(self._last_actions),
        )

    def get_action_definition(self) -> spaces.Space:
        """Return actions definition defined by the shape parametrization.

        Returns:
            spaces.Space: VariableLocations that have 'wriggle' room.
        """
        if self.discrete_actions:
            return spaces.Discrete(len(self._actions) * 2)
        else:
            return spaces.Box(low=-1, high=1, shape=(len(self._actions),))

    def get_observation_definition(self) -> Tuple[str, ObservationType]:
        """Return the geometry observation definition.

        The geometry observation by default just includes the action space of
        the shape definition.

        Returns:
            Tuple[str, ObservationType]: Tuple of the descriptor of the
                observation space and the gym based definition of the
                observation space.
        """
        if not self.action_based_observation:
            self.get_logger().warning(
                "Observation space is accessed which should not happen."
            )
        return "geometry_observation", spaces.Box(
            low=0, high=1, shape=(len(self._actions),), dtype=np.float32
        )

    def get_observation(self) -> Optional[np.ndarray]:
        """Returns the current observations.

        For the basic geometry it is possible to define the current values of
        actions to be the observations. This can be done via the variable
        `action_based_observation`.

        Returns:
            Optional[np.ndarray]: Optional action based observations.
        """
        if not self.action_based_observation:
            return None
        return np.array([
            var_loc.current_position for var_loc in self._actions
        ])

    def reset(self, validation_id: Optional[int] = None) -> Any:
        """Resets the geometry to its initial values."""
        if self.reset_with_random_action_values:
            self.apply_random_action(validation_id)
        else:
            if isinstance(self.shape_definition, list):
                for shape in self.shape_definition:
                    shape.reset()
            else:
                self.shape_definition.reset()
        return self.apply()

    def apply_random_action(self, seed: Optional[str] = None):
        """Apply a random continuous action.

        Applying a random continuous action to all movable control point
        variables. Can be activated to be used during the reset of an
        environment.

        Args:
            seed (Optional[str], optional): Seed for the generation of the
                random action. The same seed will result in always the same
                action. This functionality is chosen to make validation
                possible. If None (default) a random seed will be used and
                the action will be different each time. Defaults to None.
        """

        def _parse_string_to_int(string: str) -> int:
            chars_as_ints = [ord(char) for char in str(string)]
            string_as_int = sum(chars_as_ints)
            return string_as_int

        seed = _parse_string_to_int(str(seed)) if seed is not None else seed
        self.get_logger().debug(
            f"A random action is applied during reset with the following "
            f"seed {str(seed)}"
        )
        rng_gen = np.random.default_rng(seed)
        if self.discrete_actions:
            for action_obj in self._actions:
                action_obj.reset()
                min_bins = int(
                    (action_obj.min_value - action_obj.current_position)
                    / action_obj.step
                )
                max_bins = int(
                    (action_obj.max_value - action_obj.current_position)
                    / action_obj.step
                )
                random_action = rng_gen.integers(min_bins, max_bins)
                # Apply the random action
                action_obj.current_position += random_action * action_obj.step
        else:
            random_action = (rng_gen.random((len(self._actions),)) * 2) - 1
            for new_value, action_obj in zip(random_action, self._actions):
                action_obj.apply_continuous_action(new_value)


class FFDGeometry(Geometry):
    """FFD based variable shape."""

    mesh: MeshTypes
    export_mesh: Optional[MeshExporter] = None

    #: FreeFormDeformation used for the spline based shape optimization
    _FFD: None = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        """Geometry based on a FFD transformation.

        This geometry uses the shape parametrization to deform the given mesh,
        via a FFD.

        Raises:
            RuntimeError: FFD can only be performed via a Spline based shape.
        """
        super().__init__(**data)

        if not issubclass(type(self.shape_definition), SplineDefinition):
            raise ParserException(
                "FFDGeometry",
                "shape_definition",
                "FFD can only be performed with a splinepy Spline",
            )
        self._FFD = FFD()

    def setup(self, environment_id: UUID):
        """Setup with additional information from the environment.

        Needs to adapt export path with the environment id.

        Args:
            environment_id (UUID): Environment id.
        """
        super().setup(environment_id=environment_id)
        self._FFD.mesh = self.mesh.get_mesh()
        if self.export_mesh:
            self.export_mesh.adapt_export_path(environment_id=environment_id)

    def apply_action(self, action: Union[List[float], int]) -> Optional[Any]:
        """Function that applies a given action to the Spline.

        Args:
            action (Union[List[float], int]):  Action value depends on if the
                ActionSpace is discrete (int - Signifier of the action) or
                Continuous (List[float] - Value for each continuous variable.)
        """
        return super().apply_action(action)

    def apply(self) -> Optional[Any]:
        """Apply the shape via FFD on to the geometry.

        Returns:
            Optional[Any]: Return the correct values.
        """
        return self.apply_ffd()

    def apply_ffd(self, path: Optional[str] = None) -> Union[str, np.ndarray]:
        """Apply FFD for the current shape.

        Might move in the future to a SPORStep. Can be deactivated with
        do_not_perform_ffd.

        Apply the Free Form Deformation using the current spline to the mesh
        and export the resulting mesh to the path given.

        Args:
            path (Optional[str], optional):
                Path to where the deformed mesh was exported to or the of no
                export wanted the vertices of the mesh. Defaults to None.

        Returns:
            Union[str, np.ndarray]: Path to the exported mesh file or the
                vertices of the mesh.
        """
        self._FFD.spline = self.shape_definition.get_shape()
        # since self._FFD.mesh (this will perform the ffd) is called in the
        # export function it does not need to be called here
        if self.export_mesh:
            self.export_mesh.export_mesh(self._FFD.mesh)
            return self.export_mesh.get_export_path()
        else:
            return self._FFD.mesh.vertices


GeometryTypes = Union[Geometry, FFDGeometry]
