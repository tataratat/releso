import pathlib
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Tuple
from SbSOvRL.exceptions import SbSOvRLParserException
from SbSOvRL.util.logger import set_up_logger
from gustav import Mesh, load_mixd, load_volume_mixd
from pydantic.class_validators import root_validator
from pydantic.types import FilePath, conint

parser_logger = set_up_logger("SbSOvRL_parser")
environment_logger = set_up_logger("SbSOvRL_environment")

class Mesh(BaseModel):
    mxyz_path: Optional[FilePath] = Field(default=None, description="Please use either the path variabel xor the mxyz variable, since if used both the used mxyz path might not be the one you think.")
    mien_path: Optional[FilePath] = Field(default=None, description="Please use either the path variabel xor the mien variable, since if used both the used mien path might not be the one you think.")

    path: str # after validation Tuple["path_to_mxyz_file", "path_to_mien_file"]
    hypercube: bool = Field(description="If True Mesh is made of hypercubes. If False Mesh is made of simplexes.", default=True)
    dimensions: conint(ge=1)
    
    def get_mesh(self) -> Mesh:
        """Calls the correct method to load the mesh for the gustav library.

        Note:
            There is an error in Gustav

        Returns:
            Mesh: Mesh in gustav library format.
        """
        if self.dimensions > 2:
            environment_logger.debug(f"Loading volume mesh with mxyz file ({self.mxyz_path}) and mien file ({self.mien_path}) ...")
            mesh = load_volume_mixd(dim=self.dimensions, mxyz=self.mxyz_path, mien=self.mien_path, hexa=self.hypercube)
            environment_logger.info("Done loading volume mesh.")
        else:
            environment_logger.debug(f"Loading mesh with mxyz file ({self.mxyz_path}) and mien file ({self.mien_path}) ...")
            mesh = load_mixd(dim=self.dimensions, mxyz=self.mxyz_path, mien=self.mien_path, quad=self.hypercube)
            environment_logger.info("Done loading mesh.")
        return mesh

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
        parser_logger.debug("Validating mesh file:")
        if "mxyz_path" in values.keys() and values["mxyz_path"] is not None and "mien_path" in values.keys() and values["mien_path"] is not None:        
            parser_logger.info("Mesh files are defined by mxyz_path and mien_path.")
            return values
        elif "path" in values.keys():
            parser_logger.debug("Trying to find mxyz_path and mien_path from path variable.")
            path = pathlib.Path(values["path"]).expanduser().resolve()
            mxyz_path = None
            mien_path = None
            if path.exists():
                if path.is_dir():
                    parser_logger.debug(f"Given path is a directory, trying to find mxyz and mien files as files {path}/mxyz and {path}/mien.")
                    mxyz_path = (path / "mxyz")
                    mien_path = (path / "mien")
                    if mxyz_path.exists() and mxyz_path.is_file() and mien_path.exists() and mien_path.is_file():
                        parser_logger.debug("Mesh files located with filepath given as the root folder.")
                else: # path point to a files
                    parser_logger.debug("Given path is a file, trying to find mxyz and mien files.")
                    if path.suffix is ".mxyz":
                        mien_path = path.with_suffix(".mien")
                        mxyz_path = path
                        if mien_path.exists() and mien_path.is_file():
                            parser_logger.debug("Mesh files located with filepath given as .mxyz file.")
                        else:
                            mien_path = None
                            parser_logger.warning("Could not locate corresponding mien file with path given as .mxyz file.")
                    elif path.suffix is ".mien":
                        mxyz_path = path.with_suffix(".mxyz")
                        mien_path = path
                        if mxyz_path.exists() and mxyz_path.is_file():
                            parser_logger.debug("Mesh files located with filepath given as .mien file.")
                        else:
                            mxyz_path = None
                            parser_logger.warning("Could not locate corresponding mxyz file with path given as .mien file.")
                    elif path.name is "mxyz":
                        mxyz_path = path
                        mien_path = path.with_name("mien")
                        if mien_path.exists() and mien_path.is_file():
                            parser_logger.debug("Mesh files located with filepath given as mxyz file.")
                        else:
                            mien_path = None
                            parser_logger.warning("Could not locate corresponding mien file with path given as mxyz file.")
                    elif path.name is "mien":
                        mxyz_path = path.with_name("mxyz")
                        mien_path = path
                        if mxyz_path.exists() and mxyz_path.is_file():
                            parser_logger.debug("Mesh files located with filepath given as mien file.")
                        else:
                            mxyz_path = None
                            parser_logger.warning("Could not locate corresponding mxyz file with path given as mien file.")
                    else:
                        parser_logger.warning(f"Mesh file given as existing file {path} but could not find mxyz nor mien file.")
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
        
