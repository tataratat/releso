from nutils import mesh, function, solver, export
from nutils.expression_v2 import Namespace
from typing import Tuple
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
):
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

    return (state0, state1)


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


if __name__ == '__main__':

    geometry_tpl = setup_mesh('./2DChannelTria.msh2')
    namespace = setup_namespace(geometry_tpl)
    states_tpl = run_simulation(geometry_tpl, namespace)
    post_process(geometry_tpl, namespace, states_tpl)
