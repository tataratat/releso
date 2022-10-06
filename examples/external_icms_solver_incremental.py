import logging
import pathlib
from typing import List, Any, Dict, Optional, Tuple
# from random import randint
# from numpy.random import rand
import numpy as np
from SbSOvRL.util.logger import VerbosityLevel, set_up_logger
from SbSOvRL.util.reward_helpers import load_json, write_json, spor_com_parse_arguments


# Import user modules.
import os
import numpy as np
from meshpy import (
    mpy,
    InputFile,
    Function,
    BoundaryCondition,
    MaterialReissner,
    Mesh,
    Beam3rHerm2Line3,
)
from meshpy.mesh_creation_functions import create_beam_mesh_from_nurbs
from meshpy.header_functions import (
    set_header_static,
    set_runtime_output,
)
from meshpy.simulation_manager import Simulation, SimulationManager


def create_model(base_path, control_points_y):
    """
    Create the input file.
    """

    beam_length = 5.0
    beam_young = 1000.0
    beam_radius = 0.1

    # Setup the beam geometry.
    input_file = InputFile()

    mat = MaterialReissner(youngs_modulus=beam_young, radius=beam_radius)
    mesh = Mesh()

    control_points = [[0, 0, 0]]
    for i, control_point_y in enumerate(control_points_y):
        control_points.append(
            [(i + 1) * beam_length / len(control_points_y), control_point_y, 0.0]
        )
    knotvector = [0, 0, 0]
    for i in range(len(control_points_y) - 2):
        knotvector.append(i + 1)
    knotvector.extend([i + 2] * 3)

    # Setup the nurbs curve.
    from geomdl import NURBS

    curve = NURBS.Curve()
    curve.degree = 2
    curve.ctrlpts = control_points
    curve.knotvector = knotvector
    beam_set = create_beam_mesh_from_nurbs(mesh, Beam3rHerm2Line3, mat, curve, n_el=20, tol=1e-5)
    input_file.add(mesh)

    def get_fun(mid, wide):
        """
        Get the function of a single sinus wave.
        """

        string = (
            "(heaviside(x-{0}+{1}*1.5)*heaviside({0}+1.5*{1}-x)*cos((x-{0})*pi/{1}))"
        )
        scaling_function = "(heaviside(x-{0}+0.5*{1})*heaviside({0}+{1}*0.5-x)+1)"
        return (string + "*" + scaling_function).format(mid, wide)

    fun = Function(
        "COMPONENT 0 FUNCTION t"
    )
    input_file.add(fun)
    input_file.add(
        BoundaryCondition(
            beam_set["start"],
            "NUMDOF 9 ONOFF 1 1 1 1 1 1 0 0 0 VAL 0 0 0 0 0 0 0 0 0 FUNCT 0 0 0 0 0 0 0 0 0",
            bc_type=mpy.bc.dirichlet,
        )
    )
    input_file.add(
        BoundaryCondition(
            beam_set["line"],
            "NUMDOF 9 ONOFF 0 1 0 0 0 0 0 0 0 VAL 0 0.001 0 0 0 0 0 0 0 FUNCT 0 1 0 0 0 0 0 0 0",
            bc_type=mpy.bc.neumann,
        )
    )

    # Set headers.
    set_header_static(
        input_file, time_step=0.25, n_steps=4, tol_residuum=1e-14, tol_increment=1e-8
    )
    set_runtime_output(input_file, output_triad=False)

    input_file_path = os.path.join(base_path, "input.dat")
    input_file.write_input_file(input_file_path)
    return Simulation(input_file_path)


from vtk.util import numpy_support as VN
from vtk_utils.vtk_utils import PVDCollection


def integrate(X, u):
    """
    Evaluate the integral of the square displacements over the length of the beam.
    """

    return np.sum(np.square(X + u)[:, 1]) * 0.1


def post_process(base_dir):
    """
    Integrate the displacement of the beam..
    """

    # Check if there was an error with the simulation
    with open(os.path.join(base_dir, "xxx.log"), "r") as log_file:
        for line in log_file.readlines():
            if "PROC 0 ERROR" in line:
                logging.getLogger("SbSOvRL_rl").info(f"Error in the simulation")
                return False

    beam_result = PVDCollection(os.path.join(base_dir, "xxx-structure-beams.pvd"))
    reader = beam_result.get_time_step(beam_result.get_time_steps()[-1])
    reader.Update()
    data = reader.GetOutput()

    x = VN.vtk_to_numpy(data.GetPoints().GetData())
    u = VN.vtk_to_numpy(data.GetPointData().GetArray("displacement"))
    X = x - u

    diff = np.zeros([len(X) - 1])
    for i in range(len(X) - 1):
        diff[i] = X[i + 1, 0] - X[i, 0]

    double_entries = np.where(np.abs(diff) < 1e-10)[0]

    def delete_entries(var):
        return np.delete(var, double_entries, axis=0)

    u = delete_entries(u)
    X = delete_entries(X)

    cost_function = integrate(X, u)
    logging.getLogger("SbSOvRL_rl").info(f"Cost function: {cost_function}")
    return cost_function



def main(args, reward_solver_log) -> Dict[str, Any]:
    pathlib.Path(f"{args.run_id}").mkdir(exist_ok=True, parents=True)

    # do you need persistent local storage?
    local_variable_store_path = pathlib.Path(f"{args.run_id}/local_variable_store.json")

    # initialize/reset variable store
    if args.reset or not local_variable_store_path.exists():
        local_variable_store = {
            "last_cost": None
        }
        # initialize persistent local variable store


        write_json(local_variable_store_path, local_variable_store)
    
    # load local variables
    local_variable_store = load_json(local_variable_store_path)

    # run your code here

# expected cost function and state vector
#0.002593258178253796
#[ 0.20488688 -0.51264381 -1.16020725 -0.46927415]


    state_vector = np.array(args.json_object['info']['control_points']).flatten()
    logging.getLogger("SbSOvRL_rl").info(f"The current state vector is: {state_vector}")
    simulation = create_model(f"{args.run_id}", state_vector)
    manager = SimulationManager(f"{args.run_id}")
    manager.add(simulation)
    manager.run_simulations_and_wait_for_finish(
        baci_build_dir="/home/a11bivst/baci/work/release/"
    )
    cost_function = post_process(f"{args.run_id}")

    done = False
    info = {}
    last_cost = local_variable_store["last_cost"]
    if last_cost is None:
        reward = 0
    elif cost_function < 0.01:
        reward = 5
        done = True
        info["reset_reason"] = "converged"
    elif cost_function == False or abs(cost_function) > 10:
        reward = -10
    else:
        reward = last_cost - cost_function
        if reward < 0:
            reward *= 2
        
    local_variable_store["last_cost"] = cost_function

    return_dict = {
        "reward": reward,
        "done": done,
        "info": info,
        "observations": []
    }
    
    # store persistent local variables
    write_json(local_variable_store_path, local_variable_store)
    
    logging.getLogger("reward_solver").info(f"The current intervals values are: {return_dict}")
    return return_dict


if __name__ == "__main__":
    args = spor_com_parse_arguments()
    
    
    base_path = pathlib.Path(str(args.base_save_location))/str(args.environment_id)/str(args.run_id)
    base_path.mkdir(exist_ok=True, parents=True)
    local_variable_store_path = base_path / "local_variable_store.json"
    
    reward_solver_logger = set_up_logger("reward_solver", base_path, VerbosityLevel.INFO, console_logging=False)

    return_dict = main(args, reward_solver_logger)
    
    print(return_dict)