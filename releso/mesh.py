"""File holds definition classes for the mesh implementation."""

import pathlib
from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Literal, Optional, Union

from pydantic import Field, PrivateAttr
from pydantic.class_validators import root_validator, validator
from pydantic.types import FilePath, conint

from releso.base_model import BaseModel
from releso.exceptions import ParserException
from releso.util.logger import get_parser_logger
from releso.util.types import GustafMeshTypes

try:
    from gustaf.io import meshio, mixd
except ImportError as err:  # pragma: no cover
    from releso.util.module_import_raiser import ModuleImportRaiser

    meshio = ModuleImportRaiser("gustaf", err)
    mixd = ModuleImportRaiser("gustaf", err)


class MeshHierarchy(Enum):
    """Enum to define the mesh hierarchy."""

    Vertices = 0
    Edges = 1
    Faces = 2
    Volumes = 3


class MeshExporter(BaseModel):
    """Class which defines in which format and where the mesh is exported."""

    #: format to which the mesh should be exported to
    mesh_format: Literal["mixd"] = Field(default="mixd", alias="format")
    #: path to where the mesh should be exported to
    export_path: Union[str, pathlib.Path]

    #: internal variable if the export path was changed from a different value.
    #: This is the value is returned when the export path is queried.
    _export_path_changed: Optional[str] = PrivateAttr(default=None)

    @root_validator(pre=True)
    @classmethod
    def validate_path_has_correct_ending(cls, values) -> pathlib.Path:
        """Validator export_path.

        Validate, that the given path has the correct suffix and that the path
        exists.

        Args:
            values (Any): value to validate

        Raises:
            ParserException: If validation fails throws this error.

        Returns:
            pathlib.Path:
                Path to where the mesh should per default be exported to.
        """
        path = pathlib.Path(values.get("export_path")).expanduser().resolve()
        mesh_format = values.get("format")
        if mesh_format == "mixd":
            if path.suffix == "":
                path = path.with_name("_.xns")
            elif not path.suffix == ".xns":
                raise ParserException(
                    "MeshExporter",
                    "export_path",
                    "The format and the path suffix do not match. The format"
                    f" is defined as {mesh_format} but the suffix is"
                    f" {path.suffix}",
                )
        else:
            raise ParserException(
                "MeshExporter", "format", f"{mesh_format=} is not supported."
            )
        values["export_path"] = path
        return values

    def get_export_path(self) -> pathlib.Path:
        """Direct return of object variable.

        Returns:
            str:
                Path to the default export location of the mesh during
                environment operations. (So that the solver can use it.)
        """
        if not self._export_path_changed:
            self.adapt_export_path()
        return self._export_path_changed

    def adapt_export_path(self, environment_id: str = "0"):
        """If placeholder in export path insert environment_id into it.

        Args:
            environment_id (str): Environment ID
        """
        self._export_path_changed = pathlib.Path(
            str(self.export_path).format(environment_id)
        )
        self._export_path_changed.parent.mkdir(parents=True, exist_ok=True)
        self.get_logger().debug(
            f"Adapted mesh export path to the following value "
            f"{self._export_path_changed}."
        )

    def export_mesh(self, mesh: GustafMeshTypes, space_time: bool = False):
        """Exports the mesh.

        Args:
            mesh (GustafMeshTypes): _description_
            space_time (bool, optional): _description_. Defaults to False.

        Raises:
            RuntimeError: _description_
        """
        if self.mesh_format == "mixd":
            mixd.export(self._export_path_changed, mesh, space_time=space_time)
        else:
            raise RuntimeError(
                f"The requested format {self.mesh_format} is not supported."
            )


class Mesh(BaseModel):
    """Abstract class used to read in the mesh file and load it."""

    #: path to the mesh file (might be not used in case of mixd )
    path: Optional[Union[str, pathlib.Path]] = None
    #: Path to the default export location of the mesh during environment
    #: operations. (So that the solver can use it.)
    export: Optional[MeshExporter]
    #: Number of dimensions of the mesh.
    dimensions: conint(ge=1)

    @abstractmethod
    def get_mesh(self) -> GustafMeshTypes:
        """Calls the correct method to load the mesh for the gustaf library.

        Note:
            There is an error in a version of gustaf during the loading process
            . Please check with the maintainer of this package if you have
            trouble.

        Returns:
            gustaf.Mesh: Mesh in gustaf library format.
        """

    def adapt_export_path(self, environment_id: str):
        """If placeholder in export path insert environment_id into it.

        Args:
            environment_id (str): Environment ID
        """
        if self.export:
            self.export.adapt_export_path(environment_id=environment_id)

    def get_export_path(self) -> Optional[pathlib.Path]:
        """Direct return of object variable.

        Returns:
            str:
                Path to the default export location of the mesh during
                environment operations. (So that the solver can use it.)
        """
        if self.export:
            return self.export.get_export_path()

        self.get_logger().warning(
            (
                "No mesh exporter definition. Please define one if you want "
                "export functionality."
            ),
        )
        return None


class MixdMesh(Mesh):
    """Class used to read in the correct mixd mesh file and load it."""

    #: Please use either the path variable xor the mxyz variable, since if
    #: used both the used mxyz path might not be the one you think.
    mxyz_path: Optional[FilePath] = Field(
        default=None,
        description="Please use either the path variable xor the"
        " mxyz variable, since if used both the used mxyz path might not be "
        "the one you think.",
    )
    #: Please use either the path variable xor the mien variable, since if
    #: used both the used mien path might not be the one you think.
    mien_path: Optional[FilePath] = Field(
        default=None,
        description="Please use either the path variable xor the"
        " mien variable, since if used both the used mien path might not be "
        "the one you think.",
    )
    hypercube: bool = Field(
        description="If True Mesh is made of hypercubes. If False Mesh is made"
        " of simplexes (triangles).",
        default=True,
    )
    #: Number of dimensions of the mesh.
    dimensions: conint(ge=1)

    def get_mesh(self) -> GustafMeshTypes:
        """Calls the correct method to load the mesh for the gustaf library.

        Note:
            There is an error in a version of gustaf during the loading process
            . Please check with the maintainer of this package if you have
            trouble.

        Returns:
            gustaf.Mesh: Mesh in gustaf library format.
        """
        self.get_logger().debug(
            f"Loading volume mesh with mxyz file ({self.mxyz_path}) and "
            f"mien file ({self.mien_path}) ..."
        )
        mesh = mixd.load(
            simplex=not self.hypercube,
            volume=True if self.dimensions == 3 else False,
            mxyz=self.mxyz_path,
            mien=self.mien_path,
        )
        self.get_logger().info("Done loading mesh.")
        return mesh

    @root_validator
    @classmethod
    def validate_mxyz_mien_path(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validator for the whole class.

        Path to mxyz and mien files are correctly given. If mxyz_path and
        mien_path are given these are used. If not mxyz and mien path are
        generated from path.

        Args:
            values (Dict[str, Any]): All variables in a dict.

        Notes:
            Yes this function is overly complicated but so be it.

        Raises:
            ParserException: Error if the mien or mxyz files could not be
            found.

        Returns:
            Dict[str, Any]:
                All variables of the object. Hopefully with correct mien and
                mxyz paths
        """
        get_parser_logger().debug("Validating mesh file:")
        if (
            "mxyz_path" in values
            and values["mxyz_path"]
            and "mien_path" in values
            and values["mien_path"]
        ):
            get_parser_logger().info(
                "Mesh files are defined by mxyz_path and mien_path."
            )
            return values
        elif "path" in values.keys() and values["path"]:
            get_parser_logger().debug(
                "Trying to find mxyz_path and mien_path from path variable."
            )
            path = pathlib.Path(values["path"]).expanduser().resolve()
            mxyz_path = None
            mien_path = None
            if path.exists():
                if path.is_dir():
                    get_parser_logger().debug(
                        f"Given path is a directory, trying to find mxyz and "
                        f"mien files as files {path}/mxyz and {path}/mien."
                    )
                    mxyz_path = path / "mxyz.space"
                    mien_path = path / "mien"
                    if (
                        mxyz_path.exists()
                        and mxyz_path.is_file()
                        and mien_path.exists()
                        and mien_path.is_file()
                    ):
                        get_parser_logger().debug(
                            "Mesh files located with filepath given as the "
                            "root folder."
                        )
                else:  # path point to a files
                    get_parser_logger().debug(
                        "Given path is a file, trying to find mxyz and mien "
                        "files."
                    )
                    if path.suffix == ".mxyz":
                        mien_path = path.with_suffix(".mien")
                        mxyz_path = path
                        if mien_path.exists() and mien_path.is_file():
                            get_parser_logger().debug(
                                "Mesh files located with filepath given as "
                                ".mxyz file."
                            )
                        else:
                            mien_path = None
                            get_parser_logger().warning(
                                "Could not locate corresponding mien file with"
                                " path given as .mxyz file."
                            )
                    elif path.suffix == ".mien":
                        mxyz_path = path.with_suffix(".mxyz")
                        mien_path = path
                        if mxyz_path.exists() and mxyz_path.is_file():
                            get_parser_logger().debug(
                                "Mesh files located with filepath given as"
                                " .mien file."
                            )
                        else:
                            mxyz_path = None
                            get_parser_logger().warning(
                                "Could not locate corresponding mxyz file with"
                                " path given as .mien file."
                            )
                    elif path.name == "mxyz":
                        mxyz_path = path
                        mien_path = path.with_name("mien")
                        if mien_path.exists() and mien_path.is_file():
                            get_parser_logger().debug(
                                "Mesh files located with filepath given as "
                                "mxyz file."
                            )
                        else:
                            mien_path = None
                            get_parser_logger().warning(
                                "Could not locate corresponding mien file with"
                                " path given as mxyz file."
                            )
                    elif path.name == "mien":
                        mxyz_path = path.with_name("mxyz")
                        mien_path = path
                        if mxyz_path.exists() and mxyz_path.is_file():
                            get_parser_logger().debug(
                                "Mesh files located with filepath given as "
                                "mien file."
                            )
                        else:
                            mxyz_path = None
                            get_parser_logger().warning(
                                "Could not locate corresponding mxyz file with"
                                " path given as mien file."
                            )
                    else:
                        get_parser_logger().warning(
                            f"Mesh file given as existing file {path} but "
                            "could not find mxyz nor mien file."
                        )
                        raise ParserException(
                            "Mesh",
                            "path",
                            "Could not locate mien nor mxyz file path.",
                        )
                # If mien or mxyz file path are not complete throw error.
                if mxyz_path is None:
                    raise ParserException(
                        "Mesh", "path", "Could not locate mxyz file path."
                    )
                if mien_path is None:
                    raise ParserException(
                        "Mesh", "path", "Could not locate mien file path."
                    )
            elif path.suffix == ".xns":
                if (
                    path.with_suffix(".mxyz").exists()
                    and path.with_suffix(".mien").exists()
                ):
                    mien_path = path.with_suffix(".mxyz")
                    mxyz_path = path.with_suffix(".mien")

            if mien_path:
                values["mien_path"] = mien_path
                values["mxyz_path"] = mxyz_path
                values["path"] = mxyz_path.parent
            else:
                ParserException(
                    "MixdMesh", "path", "Could not locate mien and mxyz path"
                )
            return values
        raise ParserException(
            "Mesh",
            "[mien_|mxyz_|]path",
            "Could not locate the correct mien/mxyz paths.",
        )


class MeshIOMesh(Mesh):
    """Provides an interface to load meshio meshes.

    Since gustaf currently only really supports .msh files, this also only
    supports files with this ending. If and when gustaf implements support
    for additional files types already supported by meshio. This will be
    updated.
    """

    @validator("path")
    @classmethod
    def check_if_extension_is_supported_by_gustaf(cls, value) -> pathlib.Path:
        """Checks if the path given exists and the mesh type is supported.

        Args:
            value (Optional[str]): _description_

        Returns:
            pathlib.Path: Absolute path to the file.
        """
        if value:
            path = pathlib.Path(value).resolve().expanduser()
            if path.exists():
                if path.suffix != ".msh":
                    raise ParserException(
                        "MeshIOMesh", "path", "Mesh type not supported."
                    )
            else:
                raise ParserException(
                    "MeshIOMesh", "path", "Could not locate the mesh file."
                )

        else:
            raise ParserException(
                "MeshIOMesh",
                "path",
                "For MeshIO base mesh import a path must be provided.",
            )
        return path

    def get_mesh(self) -> GustafMeshTypes:
        """Calls the correct method to load the mesh for the gustaf library.

        Note:
            There is an error in a version of gustaf during the loading process
            . Please check with the maintainer of this package if you have
            trouble.

        Returns:
            gustaf.Mesh: Mesh in gustaf library format.
        """
        self.get_logger().debug(f"Loading mesh from path ({self.path}).")
        mesh = meshio.load(fname=self.path)
        final_mesh = None
        if len(mesh) > 1:
            for me in mesh:
                if final_mesh is None:
                    final_mesh = me
                    continue
                if (
                    MeshHierarchy[me.__class__.__name__].value
                    > MeshHierarchy[final_mesh.__class__.__name__].value
                ):
                    final_mesh = me
        else:
            final_mesh = mesh
        self.get_logger().info("Done loading mesh.")
        return self.delete_dimension_if_possible(final_mesh)

    def delete_dimension_if_possible(self, mesh: GustafMeshTypes):
        """Looks into the mesh and checks if the dimensionality can be reduced.

        If the bounds of one or more dimensions are equal, the dimension is
        removed from the mesh.

        Args:
            mesh (GustafMeshTypes): _description_
        """
        keep_dims = [0, 1, 2]
        if self.dimensions < mesh.vertices.shape[1]:
            bounds = mesh.bounds()
            for idx, (lower, upper) in enumerate(zip(*bounds)):
                if lower == upper:
                    self.get_logger().info(
                        f"Found empty dimension {idx}. Will remove it now."
                    )
                    keep_dims.pop(idx)
                    mesh.vertices = mesh.vertices[:, keep_dims]
        return mesh


MeshTypes = Union[MeshIOMesh, MixdMesh]
