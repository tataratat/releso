"""SPOR step python file making cnn observations based on xns results.

This file is not tested due to it being specific to the xns use case.
Please check if any updates to the required files are needed before
using these functions.
"""

import datetime
import pathlib
from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from releso.observation import ObservationDefinitionMulti
from releso.util.load_binary import read_mixd_double
from releso.util.plotting import get_tricontour_solution


def define_observation_definition() -> ObservationDefinitionMulti:
    """Returns the observation definition for the SPORObject.

    This is a image based observation creation routine. This means, that the
    observation created is of type CNN and has as shape of (3, 200, 200).

    Returns:
        ObservationDefinitionMulti: _description_
    """
    return ObservationDefinitionMulti(
        save_location="",  # Dummy safe location
        name="cnn_observation",
        value_min=0,
        value_max=255,
        observation_shape=(3, 200, 200),
        value_type="CNN",
    )


def get_visual_representation(
    connectivity: np.ndarray,
    sol_len: int = 3,
    height: int = 10,
    width: int = 10,
    dpi: int = 20,
) -> np.ndarray:
    """Return an array representing the calculated solution.

    This function is used to create the array which is used if the
    solution/cnn representation of the base observations are selected.

    Note:
        The calculated solution can only be found if the
        :py:class:`SporObject` of the solver is called main_solver and uses
        the xns solver.
    #TODO make it broader in its application e.g. other solvers?

    Args:
        connectivity (np.ndarray): Connectivity array of the mesh.
        sol_len (int, optional): Number of variables to return per data
            point, input can be more but not less. Assumed is u,v,p.
            Defaults to 3.
        height (int, optional): Height of the image array in inches.
            Defaults to 10.
        width (int, optional): Width of the image array in inches.
            Defaults to 10.
        dpi (int, optional): DPI of the image array. Together with the
            height and width, this variable defines the shape of the
            return array shape=(height*dpi, width*dpi, sol_len).
            Defaults to 20.

    Raises:
        RuntimeError: could not find solution file, mesh is of the
            incorrect type (needs to be triangular), mesh is of the
            incorrect dimensions (needs to be dim=2)
    """
    # if self.mesh.hypercube:
    #     raise RuntimeError(
    #         "The given mesh is not a mesh with triangular entities. "
    #         "This function was designed to work with only those.")
    # if not self.mesh.dimensions == 2:
    #     raise RuntimeError(
    #         "The given mesh has a dimension unequal two. This function "
    #         "was designed to work only with meshes of dimension two.")
    solution = read_mixd_double("ins.out", 3)
    coordinates = (
        np.fromfile("mesh/mxyz", dtype=">d").astype(np.float64).reshape(-1, 2)
    )

    # Plotting and creating resulting array
    limits_max = [1, 1, 0.2e8]
    limits_min = [-1, -1, -0.2e8]

    return get_tricontour_solution(
        width,
        height,
        dpi,
        coordinates,
        connectivity,
        solution,
        sol_len,
        limits_min,
        limits_max,
    )


def save_current_solution_as_png(
    save_location: Union[pathlib.Path, str],
    include_pressure: bool = True,
    height: int = 10,
    width: int = 10,
    dpi: int = 400,
):
    """Save the current solver solution as an image at the given location.

    Additional parameters for size can be used.

    TODO additional SPORStep for this functionality?

    Args:
        save_location (Union[pathlib.Path, str]): Where the image is to
            be saved to. Please add solution specific path, else the file
            will be overwritten if multiple solution are saved.
        include_pressure (bool, optional): Not only use x-,y-velocity
            fields but also the pressure field. Defaults to True.
        height (int, optional): Self explanatory. Defaults to 10.
        width (int, optional): Self explanatory. Defaults to 10.
        dpi (int, optional): Self explanatory. Defaults to 400.
    """
    image_arr = get_visual_representation(
        sol_len=3 if include_pressure else 2,
        width=height,
        height=width,
        dpi=dpi,
    )
    meta_data_dict = {
        "Author": "Clemens Fricke",
        "Software": "ReLeSO",
        "Creation Time": str(datetime.datetime.now()),
        "Description": "This is a description",
    }
    if isinstance(save_location, str):
        save_location = pathlib.Path(save_location)
    save_location.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(save_location, image_arr, metadata=meta_data_dict)


def main(args, logger, func_data) -> Tuple[Dict[str, Any], Any]:
    """Function which is called from the spor step.

    The parameters need to conform to the SPOR_COMM interface defined by
    ReLeSO.

    Args:
        args (_type_): _description_
        logger (_type_): _description_
        func_data (_type_): _description_

    Returns:
        Set[Dict[str, Any], Any]: _description_
    """
    # first time initialization
    if func_data is None:
        func_data = {}
        func_data["connectivity"] = (
            (np.fromfile("mesh/mien", dtype=">i") - 1)
            .astype(np.int64)
            .reshape(-1, 3)
        )

    # Loading the correct data and checking if correct attributes are set.
    cnn_observation = get_visual_representation(
        connectivity=func_data["connectivity"]
    )

    return_dict = {
        "reward": 0,
        "done": False,
        "info": {},
        "observations": [cnn_observation.T],
    }

    return return_dict, func_data
