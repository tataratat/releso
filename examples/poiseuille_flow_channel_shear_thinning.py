from nutils import mesh, function, solver, export
from nutils.expression_v2 import Namespace
from typing import Tuple, Dict, Any, Set
import numpy as np


def setup_mesh(mesh_path : str):
    """ Load a gmsh mesh from file.

    Parameters
    ----------
    mesh_path : :class:`str`
        String containg the path to the mesh file.

    Returns
    -------
    domain : :class:`nutils.topology.SimplexTopology`
        Topology of the parsed gmsh file.
    geom : :class:`nutils.function.Array`
        Mesh mapping (product of mesh nodes and basis functions).
    """
    # create the geometry and the mesh from a gmsh file
    domain, geom = mesh.gmsh(mesh_path)

    # with mesh_path as f:
    #     mesh_data = mesh.parsegmsh(f)
    
    # domain, geom =  mesh.simplex(name='gmsh', **mesh_data)

    # # TODO Change coordinates
    # coords = mesh_data['coords']
    # # overwrite geometry with new coordinates
    # geom = (domain.basis()[:, np.newaxis] * coords).sum(0)

    return (domain, geom)


def setup_namespace(
    geometry_tpl : Tuple, 
    degree : int = 2,
    A : float = 6589, 
    B : float = 0.138, 
    C : float = 0.725
):
    """ Set up the namespace for this problem.

    Parameters
    ----------
    geometry_tpl : :class:`Tuple`
        Nutils domain and geometry for this problem.
    degree : :class:`int`
        Polynomial degree of the interpolation functions for the velocity.
    A : :class:`float`
        Zero-shear viscosity.
    B : :class:`float`
        Reciprocal transition rate.
    C : :class:`float`
        Slope of viscosity curve in pseudoplastic region.

    Returns
    -------
    ns : :class:`nutils.expression_v2.Namespace`
        Namespace object for this problem.
    """
    domain, geom = geometry_tpl
    # define all symbols for this problem
    ns = Namespace()
    ns.delta = function.eye(domain.ndims)
    ns.sigma = function.ones([domain.ndims])
    ns.A = A
    ns.B = B
    ns.C = C
    ns.x = geom
    ns.define_for('x', gradient='grad', normal='n', jacobians=('dV', 'dS'))
    ns.ubasis = domain.basis('std', degree=degree).vector(domain.ndims)
    ns.pbasis = domain.basis('std', degree=degree-1)
    ns.u = function.dotarg('u', ns.ubasis)
    ns.p = function.dotarg('p', ns.pbasis)
    ns.epsilon_ij = '0.5 (grad_j(u_i) + grad_i(u_j))'
    ns.gammaDot = 'sqrt(2 epsilon_ij epsilon_ij)'
    ns.eta = 'A / ((1 + B gammaDot)^(C))'
    ns.stressNewton_ij = '2 A epsilon_ij - p delta_ij'
    ns.stressYasuda_ij = '2 eta epsilon_ij - p delta_ij'

    return ns


def run_simulation(
    geometry_tpl : Tuple, 
    namespace : Namespace, 
    intrpl_dgr : int = 4
) -> Tuple(np.ndarray, np.ndarray):
    '''
    Simulate the shear-thinning Stokes flow in a contracting channel geometry.

    Parameters
    ----------
    geometry_tpl : :class:`Tuple`
        Nutils domain and geometry for this problem.
    namespace : :class:`nutils.expression_v2.Namespace`
        Namespace object with all definitions for this problem.
    intrpl_dgr : :class:`int`
        Interpolation degree for numerical integration.
    '''

    ########
    # MESH #
    ########

    domain, geom = geometry_tpl

    #############
    # NAMESPACE #
    #############

    ns = namespace

    #######################
    # BOUNDARY CONDITIONS #
    #######################

    # inflow boundary condition (const. x-velocity of 0.5)
    usqr = domain.boundary['Inflow'].integral('(u_0 - 0.5)^2 dS' @ ns, degree=intrpl_dgr)
    usqr += domain.boundary['Inflow'].integral('(u_1)^2 dS' @ ns, degree=intrpl_dgr)

    # no-slip condition on bottom and top wall
    usqr += domain.boundary['LowerWall'].integral('u_k u_k dS' @ ns, degree=intrpl_dgr)
    usqr += domain.boundary['UpperWall'].integral('u_k u_k dS' @ ns, degree=intrpl_dgr)

    # zero vertical velocity at outflow
    usqr += domain.boundary['Outflow1'].integral('(u_1)^2 dS' @ ns, degree=intrpl_dgr)
    usqr += domain.boundary['Outflow2'].integral('(u_1)^2 dS' @ ns, degree=intrpl_dgr)
    usqr += domain.boundary['Outflow3'].integral('(u_1)^2 dS' @ ns, degree=intrpl_dgr)
    
    # compose contraints (boundary conditions) for the solver
    ucons = solver.optimize('u', usqr, droptol=1e-15)
    cons = dict(u=ucons)

    ####################
    # WEAK FORMULATION #
    ####################

    # domain integrals for Stokes problem
    uresNewton = domain.integral('grad_j(ubasis_ni) stressNewton_ij dV' @ ns, degree=intrpl_dgr)
    uresYasuda = domain.integral('grad_j(ubasis_ni) stressYasuda_ij dV' @ ns, degree=intrpl_dgr)
    pres = domain.integral('pbasis_n grad_k(u_k) dV' @ ns, degree=intrpl_dgr)

    #########
    # SOLVE #
    #########

    # Solve the linear system as initial guess for nonlinear system
    state0 = solver.solve_linear(('u', 'p'), (uresNewton, pres), constrain=cons)
    # Solve the nonlinear system
    state1 = solver.newton(('u', 'p'), (uresYasuda, pres), arguments=state0, constrain=cons).solve(tol=1e-10)

    ##############
    # EVALUATION #
    ##############

    # area patch 1
    area_patch_1 = domain.boundary['Outflow1'].integral('1 dS' @ ns, degree=intrpl_dgr)
    area_1 = area_patch_1.eval()
    # mass flow patch 1
    outflow_patch_1 = domain.boundary['Outflow1'].integral('u_k n_k dS' @ ns, degree=intrpl_dgr)
    mass_flow_1 = outflow_patch_1.eval(**state1)

    # area patch 2
    area_patch_2 = domain.boundary['Outflow2'].integral('1 dS' @ ns, degree=intrpl_dgr)
    area_2 = area_patch_2.eval()
    # mass flow patch 2
    outflow_patch_2 = domain.boundary['Outflow2'].integral('u_k n_k dS' @ ns, degree=intrpl_dgr)
    mass_flow_2 = outflow_patch_2.eval(**state1)

    # area patch 3
    area_patch_3 = domain.boundary['Outflow3'].integral('1 dS' @ ns, degree=intrpl_dgr)
    area_3 = area_patch_3.eval()
    # mass flow patch 1
    outflow_patch_3 = domain.boundary['Outflow3'].integral('u_k n_k dS' @ ns, degree=intrpl_dgr)
    mass_flow_3 = outflow_patch_3.eval(**state1)

    col1 = 'RNG'
    col2 = 'area'
    col3 = 'mass flow'
    print()
    print(f'{col1:>9} {col2:>15} {col3:>15}')
    print(f'Outflow 1 {area_1:>15.7f} {mass_flow_1:>15.7f}')
    print(f'Outflow 2 {area_2:>15.7f} {mass_flow_2:>15.7f}')
    print(f'Outflow 3 {area_3:>15.7f} {mass_flow_3:>15.7f}')
    print()

    return (
        np.array([area_1,area_2,area_3]),
        np.array([mass_flow_1, mass_flow_2, mass_flow_3]))


def post_process(geometry_tpl : Tuple, ns : Namespace, states_tpl : Tuple):
    """ Post-process the simulation results.

    Parameters
    ----------
    geometry_tpl : :class:`Tuple`
        Nutils domain and geometry for this problem.
    namespace : :class:`nutils.expression_v2.Namespace`
        Namespace object with all definitions for this problem.
    states_tpl : :class:`Tuple`
        Tuple containing the Nutils simulation states of the linear and the 
        nonlinear problem.
    """
    domain, geom = geometry_tpl
    state0, state1 = states_tpl

    bezier = domain.sample('bezier', 9)
    x, u, p, eta = bezier.eval([ns.x, ns.u, ns.p, ns.eta], **state1)
    export.triplot('stokes_u0.png', x, u[:,0], tri=bezier.tri, hull=bezier.hull)
    export.triplot('stokes_u1.png', x, u[:,1], tri=bezier.tri, hull=bezier.hull)
    export.triplot('stokes_p.png', x, p, tri=bezier.tri, hull=bezier.hull)
    export.triplot('stokes_eta.png', x, eta, tri=bezier.tri, hull=bezier.hull)

def compute_quality_criterion(
    area : np.ndarray,
    mass_flow : np.ndarray) -> Tuple(np.ndarray, np.ndarray):

    # compute total mass flow over and area of outflow boundary
    totalMassFlow = mass_flow.sum()
    totalArea = area.sum()
    # compute the average velocity at the outflow
    averageVelocity = totalMassFlow / totalArea

    localVelocity = mass_flow / area
    ratio = localVelocity / averageVelocity
    qualityCriterion = (ratio-1) / np.max(
        np.hstack(ratio, np.ones_like(ratio)), axis=1)

    return ratio, qualityCriterion

def calculate_reward(
    quality_criterions: np.ndarray,
    last_quality_criterions: np.ndarray, **kwargs) -> float:
    """Function that actually calculates the reward.

    Note: Currently the reward is calculated the same way Michael calculated it.

    Args:
        quality_criterions (List[float]): Current quality criterion values.
        last_quality_criterions (Optional[List[float]], optional): To track progress the last quality criterions is also an input. Defaults to None.

    Returns:
        float: Calculated reward
    """
    acceptance_value: float = 0.358
    info = {}
    reward = 0
    quality_reward_sum = (quality_criterions**2).sum()
    last_quality_reward_sum =  (last_quality_criterions**2).sum()
    done = False
    
    if quality_reward_sum < acceptance_value:   # achieved goal quality 
        reward = 5
        done = True
        info["reset_reason"] = "goal_achieved"
    else:   # last_quality_reward_sum >= quality_reward_sum result got worse relative to last step
        reward = quality_reward_sum * -1.0
    return reward, info, done

def main(args, logger, func_data) -> Set[Dict[str, Any], Any]:
    mesh_path = './2DChannelTria.msh2'

    # first time initialization
    if func_data is None:
        func_data = dict()
    
    if "domain" not in func_data.keys():
        func_data["domain"], func_data["geom"] = setup_mesh(
            mesh_path=mesh_path)
        func_data["basis"] = func_data["domain"].basis("std", degree=1).vector(
            func_data["domain"].ndims)
    
    # adapt geometry
    #TODO  args.json_object['info']['mesh_coords']
    coords = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])

    func_data["geom"] = (func_data["basis"][:, np.newaxis] * coords).sum(0)

    namespace = setup_namespace((func_data["domain"],func_data["geom"]))

    ratio, quality_criterion = compute_quality_criterion(*run_simulation(
        (func_data["domain"],func_data["geom"]), namespace
    ))
    
    #first time or on reset
    if "last_quality_criterion" not in func_data.keys() or args.reset:
        func_data["last_quality_criterion"] = quality_criterion

    reward, info, done = calculate_reward(
        quality_criterion, func_data["last_quality_criterion"])

    # overwrite old quality_criterion
    func_data["last_quality_criterion"] = quality_criterion

    return_dict = {
        "reward": reward,
        "done": done,
        "info": info,
        "observations": [
            quality_criterion.tolist()
        ]
    }

    return return_dict, func_data

if __name__ == '__main__':

    geometry_tpl = setup_mesh('./2DChannelTria.msh2')
    namespace = setup_namespace(geometry_tpl)
    states_tpl = run_simulation(geometry_tpl, namespace)
    post_process(geometry_tpl, namespace, states_tpl)
