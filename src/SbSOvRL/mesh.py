import pathlib
from pydantic import Field
from typing import Optional, Dict, Any
from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.util.logger import get_parser_logger
from gustav import Mesh, load_mixd, load_volume_mixd
from pydantic.class_validators import root_validator, validator
from pydantic.types import FilePath, conint
from SbSOvRL.base_model import SbSOvRL_BaseModel

class Mesh(SbSOvRL_BaseModel):
    """
    Class used to read in the correct base mesh file and load it.
    """
    mxyz_path: Optional[FilePath] = Field(default=None, description="Please use either the path variabel xor the mxyz variable, since if used both the used mxyz path might not be the one you think.") #: Please use either the path variabel xor the mxyz variable, since if used both the used mxyz path might not be the one you think. 
    mien_path: Optional[FilePath] = Field(default=None, description="Please use either the path variabel xor the mien variable, since if used both the used mien path might not be the one you think.") #: Please use either the path variabel xor the mien variable, since if used both the used mien path might not be the one you think.

    path: str   #: after validation Tuple["path_to_mxyz_file", "path_to_mien_file"]
    export_path: str    #: Path to the default export location of the mesh during environment operations. (So that the solver can use it.)
    hypercube: bool = Field(description="If True Mesh is made of hypercubes. If False Mesh is made of simplexes (triangles).", default=True)    #: If True Mesh is made of hypercubes. If False Mesh is made of simplexes (triangles).
    dimensions: conint(ge=1)    #: Number of dimensions of the mesh.

    @validator("export_path")
    @classmethod
    def validate_path_has_correct_ending(cls, v) -> pathlib.Path:
        """Validate that the given path has a xns suffix or if no suffix xns as name. Also converts str to pathlib.Path

        Args:
            v ([type]): value to validate

        Raises:
            SbSOvRLParserException: If validation fails throws this error.

        Returns:
            pathlib.Path: Path to where the mesh should per default be exported to.
        """
        path = pathlib.Path(v).expanduser().resolve()
        if path.suffix == '':
            if path.name == "xns":
                path = path.with_name("_.xns")
            else:
                raise SbSOvRLParserException("Mesh", "export_path", "Currently only the name xns without suffix is supported. If campiga should be added pls notify the author.")
        elif not path.suffix == ".xns":
            raise SbSOvRLParserException("Mesh", "export_path", "Currently only the suffix xns is supported. If campiga should be added pls notify the author.")
        return path
        
    def get_mesh(self) -> Mesh:
        """Calls the correct method to load the mesh for the gustav library.

        Note:
            There is an error in a version of gustav during the loading process. Please check with the maintainer of this package if you have trouble.

        Returns:
            gustav.Mesh: Mesh in gustav library format.
        """
        if self.dimensions > 2:
            self.get_logger().debug(f"Loading volume mesh with mxyz file ({self.mxyz_path}) and mien file ({self.mien_path}) ...")
            mesh = load_volume_mixd(dim=self.dimensions, mxyz=self.mxyz_path, mien=self.mien_path, hexa=self.hypercube)
            self.get_logger().info("Done loading volume mesh.")
        else:
            self.get_logger().debug(f"Loading mesh with mxyz file ({self.mxyz_path}) and mien file ({self.mien_path}) ...")
            mesh = load_mixd(dim=self.dimensions, mxyz=self.mxyz_path, mien=self.mien_path, quad=self.hypercube)
            self.get_logger().info("Done loading mesh.")
        return mesh

    def get_export_path(self) -> str:
        """Direct return of object variable.

        Returns:
            str: Path to the default export location of the mesh during environment operations. (So that the solver can use it.)
        """
        return self.export_path


    @root_validator
    @classmethod
    def validate_mxyz_mien_path(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Path to mxyz and mien files are correctly given. If mxyz_path and mien_path are given these are used. If not mxyz and mien path are generated from path.

        Args:
            values (Dict[str, Any]): All variables in a dict.

        Notes:
            Yes this function is overly complicated but so be it.

        Raises:
            SbSOvRLParserException: Error if the mien or mxyz files could not be found.

        Returns:
            Dict[str, Any]: All variables of the object. Hopefully with correct mien and mxyz paths
        """
        get_parser_logger().debug("Validating mesh file:")
        if "mxyz_path" in values.keys() and values["mxyz_path"] is not None and "mien_path" in values.keys() and values["mien_path"] is not None:        
            get_parser_logger().info("Mesh files are defined by mxyz_path and mien_path.")
            return values
        elif "path" in values.keys():
            get_parser_logger().debug("Trying to find mxyz_path and mien_path from path variable.")
            path = pathlib.Path(values["path"]).expanduser().resolve()
            mxyz_path = None
            mien_path = None
            if path.exists():
                if path.is_dir():
                    get_parser_logger().debug(f"Given path is a directory, trying to find mxyz and mien files as files {path}/mxyz and {path}/mien.")
                    mxyz_path = (path / "mxyz.space")
                    mien_path = (path / "mien")
                    if mxyz_path.exists() and mxyz_path.is_file() and mien_path.exists() and mien_path.is_file():
                        get_parser_logger().debug("Mesh files located with filepath given as the root folder.")
                else: # path point to a files
                    get_parser_logger().debug("Given path is a file, trying to find mxyz and mien files.")
                    if path.suffix == ".mxyz":
                        mien_path = path.with_suffix(".mien")
                        mxyz_path = path
                        if mien_path.exists() and mien_path.is_file():
                            get_parser_logger().debug("Mesh files located with filepath given as .mxyz file.")
                        else:
                            mien_path = None
                            get_parser_logger().warning("Could not locate corresponding mien file with path given as .mxyz file.")
                    elif path.suffix == ".mien":
                        mxyz_path = path.with_suffix(".mxyz")
                        mien_path = path
                        if mxyz_path.exists() and mxyz_path.is_file():
                            get_parser_logger().debug("Mesh files located with filepath given as .mien file.")
                        else:
                            mxyz_path = None
                            get_parser_logger().warning("Could not locate corresponding mxyz file with path given as .mien file.")
                    elif path.name == "mxyz":
                        mxyz_path = path
                        mien_path = path.with_name("mien")
                        if mien_path.exists() and mien_path.is_file():
                            get_parser_logger().debug("Mesh files located with filepath given as mxyz file.")
                        else:
                            mien_path = None
                            get_parser_logger().warning("Could not locate corresponding mien file with path given as mxyz file.")
                    elif path.name == "mien":
                        mxyz_path = path.with_name("mxyz")
                        mien_path = path
                        if mxyz_path.exists() and mxyz_path.is_file():
                            get_parser_logger().debug("Mesh files located with filepath given as mien file.")
                        else:
                            mxyz_path = None
                            get_parser_logger().warning("Could not locate corresponding mxyz file with path given as mien file.")
                    else:
                        get_parser_logger().warning(f"Mesh file given as existing file {path} but could not find mxyz nor mien file.")
                        SbSOvRLParserException("Mesh", "path", "Could not locate mien nor mxyz file path.")
                # If mien or mxyz file path are not complete throw error.
                if mxyz_path is None:
                    SbSOvRLParserException("Mesh", "path", "Could not locate mxyz file path.")
                if mien_path is None:
                    SbSOvRLParserException("Mesh", "path", "Could not locate mien file path.")
            values["mien_path"] = mien_path
            values["mxyz_path"] = mxyz_path                
            return values
        raise SbSOvRLParserException("Mesh", "[mien_|mxyz_|]path", "Could not locate the correct mien/mxyz paths.")
        
