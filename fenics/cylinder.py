import sys
import time
import logging

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl

import dolfinx
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from dolfinx.io import gmshio, XDMFFile

from vars import *


logger = logging.getLogger(__name__)


def log(msg):
    print(msg, flush=True)
    logger.info(msg)


def get_mesh(fname, comm):
    mesh, cell_markers, facet_markers = gmshio.read_from_msh(
        fname + ".msh", comm, rank=0, gdim=dim3
    )
    return mesh, cell_markers, facet_markers


def top_circle(x):
    center_x = 0.0
    center_y = 0.0
    radius = 0.4
    distance = np.sqrt((x[0] - center_x) ** 2 + (x[1] - center_y) ** 2)
    return distance, radius


def top_boundary(x):
    distance, radius = top_circle(x)
    return np.logical_and(distance <= radius, np.isclose(x[2], l))


def initial_condition(x):
    distance, radius = top_circle(x)
    return np.where(np.logical_and(distance <= radius, np.isclose(x[2], l)), 1.0, 0.0)


def get_velocity_field(mesh):
    # v = np.random.uniform(low=-0.5, high=0.5, size=(3,))
    # v[2] = -v[2] if v[2] >= 0 else v[2]
    # velocity = Constant(mesh, PETSc.ScalarType((v[0], v[1], v[2])))
    velocity = Constant(mesh, PETSc.ScalarType((0.0, 0.0, -0.2)))

    # V = functionspace(mesh, ("DG", 0, (dim3,)))
    # velocity = Function(V)
    # x = V.tabulate_dof_coordinates()
    # xy_vec = np.zeros((x.shape[0], 2))
    # z_vec = np.full((x.shape[0], 1), -0.4)
    # z_vec = np.random.uniform(low=-0.5, high=-0.1, size=(x.shape[0], 1))
    # velocity.x.array[:] = np.hstack([xy_vec, z_vec]).flatten()

    return velocity


def get_diffusivity_field(mesh):
    diffusivity = Constant(mesh, PETSc.ScalarType(0.02))

    # V = functionspace(mesh, ("DG", 0))
    # diffusivity = Function(V)
    # x = V.tabulate_dof_coordinates()
    # diffusivity.x.array[:] = np.random.uniform(0.01, 0.02, x.shape[0])

    return diffusivity


def write_mesh(file, fname, mesh, cell_markers, facet_markers):
    mesh.name = "mesh"
    cell_markers.name = f"{mesh.name}_cells"
    facet_markers.name = f"{mesh.name}_facets"

    mesh.topology.create_connectivity(2, 3)
    file.write_mesh(mesh)
    file.write_meshtags(
        cell_markers,
        mesh.geometry,
        geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry",
    )
    file.write_meshtags(
        facet_markers,
        mesh.geometry,
        geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{mesh.name}']/Geometry",
    )


def solve(mesh, fname, cond_type, cell_markers, facet_markers):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Define function space
    V = functionspace(mesh, ("Lagrange", 1))

    # Initial condition
    s = time.time()
    c_n = Function(V)
    if cond_type == "ic":
        c_n.interpolate(initial_condition)
    if cond_type == "bc":
        c_top_bc = 1.0
        top_bc_tag = 2
        bc_dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facet_markers.find(top_bc_tag))

        top_bc = dirichletbc(PETSc.ScalarType(c_top_bc), bc_dofs, V)

        c_n.interpolate(lambda x: np.zeros_like(x[0]))
        # for dof in bc_dofs:
        c_n.x.array[bc_dofs] = c_top_bc
    e = time.time()
    if rank == 0:
        log(f"2: Took {e-s:.4f}s")

    # Velocity and diffusivity
    s = time.time()
    velocity = get_velocity_field(mesh)
    diffusivity = get_diffusivity_field(mesh)
    e = time.time()
    if rank == 0:
        log(f"3: Took {e-s:.4f}s")

    # Define test and trial functions
    c = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Time step size
    dt = 0.01

    # Define variational forms
    a = (
        c * v * ufl.dx
        + dt
        * (ufl.dot(velocity, ufl.grad(c)) * v + diffusivity * ufl.dot(ufl.grad(c), ufl.grad(v)))
        * ufl.dx
    )
    L = c_n * v * ufl.dx

    # Create linear problem
    s = time.time()
    petsc_options = {"ksp_type": "cg", "ksp_rtol": 1e-6, "ksp_atol": 1e-10, "ksp_max_it": 1000}
    if cond_type == "ic":
        problem = LinearProblem(a, L, petsc_options=petsc_options)
        # problem = LinearProblem(a, L)
    if cond_type == "bc":
        problem = LinearProblem(a, L, bcs=[top_bc], petsc_options=petsc_options)
        # problem = LinearProblem(a, L, bcs=[top_bc])
    e = time.time()
    if rank == 0:
        log(f"4: Took {e-s:.4f}s")

    # Time-stepping
    T = 5.0
    t = 0.0
    c = Function(V)

    # Solve and save
    s = time.time()
    with XDMFFile(mesh.comm, f"{fname}_results_{cond_type}.xdmf", "w") as file:
        write_mesh(file, fname, mesh, cell_markers, facet_markers)

        diffusivity.name = "diffusivity"
        file.write_function(diffusivity)
        velocity.name = "velocity"
        file.write_function(velocity)
        c_n.name = "concentration"
        file.write_function(c_n)

        while t < T:
            t += dt
            if rank == 0:
                log(f"t = {t:.2f} / T = {T:.2f}")

            si = time.time()
            # Solve the linear problem
            c = problem.solve()
            ei = time.time()
            if rank == 0:
                log(f"5: Took {ei-si:.4f}s")

            # Update the previous solution
            c_n.x.array[:] = c.x.array[:]

            # Write solution to file
            file.write_function(c_n, t)
    e = time.time()
    if rank == 0:
        log(f"6: Took {e-s:.4f}s")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if len(sys.argv) == 3:
        logging.basicConfig(
            filename=f"{sys.argv[1]}_{sys.argv[2]}.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s  %(message)s",
            datefmt="%d %B %Y %H:%M:%S",
        )

        s = time.time()
        mesh, cell_markers, facet_markers = get_mesh(sys.argv[1], comm)
        e = time.time()
        if rank == 0:
            log(f"1: Took {e-s:.4f}s")

        if sys.argv[2] == "ic":
            solve(
                mesh,
                fname=sys.argv[1],
                cond_type="ic",
                cell_markers=cell_markers,
                facet_markers=facet_markers,
            )
        elif sys.argv[2] == "bc":
            solve(
                mesh,
                fname=sys.argv[1],
                cond_type="bc",
                cell_markers=cell_markers,
                facet_markers=facet_markers,
            )
        else:
            print("Wrong condition.")
    else:
        print("Must have filename.")
