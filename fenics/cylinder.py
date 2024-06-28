import sys
import time

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
)
from dolfinx.io import XDMFFile

from vars import *


def get_mesh(fname):
    with XDMFFile(MPI.COMM_WORLD, fname + ".xdmf", "r") as file:
        mesh = file.read_mesh(name="mesh")
    return mesh


def top_boundary(x):
    center_x = 0.0
    center_y = 0.0
    radius = 0.4

    distance = np.sqrt((x[0] - center_x) ** 2 + (x[1] - center_y) ** 2)
    return np.logical_and(distance <= radius, np.isclose(x[2], l))


def initial_condition(x):
    center_x = 0.0
    center_y = 0.0
    radius = 0.4

    distance = np.sqrt((x[0] - center_x) ** 2 + (x[1] - center_y) ** 2)
    # Set initial condition to 1.0 within the circle at the top
    return np.where(np.logical_and(distance <= radius, np.isclose(x[2], l)), 1.0, 0.0)


def get_velocity_field(mesh):
    # v = np.random.uniform(low=-0.5, high=0.5, size=(3,))
    # v[2] = -v[2] if v[2] >= 0 else v[2]
    # velocity = Constant(mesh, PETSc.ScalarType((v[0], v[1], v[2])))
    velocity = Constant(mesh, PETSc.ScalarType((0.0, 0.0, -0.2)))

    # V = functionspace(mesh, ("DG", 0, (gdim,)))
    # velocity = Function(V)
    # x = V.tabulate_dof_coordinates()
    # xy_vec = np.random.uniform(low=-0.2, high=0.2, size=(x.shape[0], 2))
    # z_vec = np.full((x.shape[0], 1), -0.4)
    # velocity.x.array[:] = np.hstack([xy_vec, z_vec]).flatten()

    return velocity


def get_diffusivity_field(mesh):
    # diffusivity = Constant(mesh, PETSc.ScalarType(0.01))

    V = functionspace(mesh, ("DG", 0))
    diffusivity = Function(V)
    x = V.tabulate_dof_coordinates()
    diffusivity.x.array[:] = np.random.uniform(0.01, 0.02, x.shape[0])

    return diffusivity


def solve(mesh, fname, cond_type):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Define function space
    V = functionspace(mesh, ("Lagrange", 1))

    # Initial condition
    s = time.time()
    if cond_type == "ic":
        c_n = Function(V)
        c_n.interpolate(initial_condition)
    if cond_type == "bc":
        top_bc = dirichletbc(PETSc.ScalarType(1.0), locate_dofs_geometrical(V, top_boundary), V)
        c_n = Function(V)
        c_n.interpolate(lambda x: np.zeros_like(x[0]))
    e = time.time()
    if rank == 0:
        print(f"2: Took {e-s:.4f}s", flush=True)

    # Velocity and diffusivity
    s = time.time()
    velocity = get_velocity_field(mesh)
    diffusivity = get_diffusivity_field(mesh)
    e = time.time()
    if rank == 0:
        print(f"3: Took {e-s:.4f}s", flush=True)

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
        print(f"4: Took {e-s:.4f}s", flush=True)

    # Time-stepping
    T = 5.0
    t = 0.0
    c = Function(V)

    # Solve and save
    s = time.time()
    with XDMFFile(mesh.comm, fname + "_results.xdmf", "w") as file:
        file.write_mesh(mesh)
        diffusivity.name = "diffusivity"
        file.write_function(diffusivity)

        while t < T:
            t += dt
            print(f"t = {t:.2f} / T = {T:.2f}")

            si = time.time()
            # Solve the linear problem
            c = problem.solve()
            ei = time.time()
            if rank == 0:
                print(f"5: Took {ei-si:.4f}s", flush=True)

            # Update the previous solution
            c_n.x.array[:] = c.x.array[:]

            # Write solution to file
            file.write_function(c, t)
    e = time.time()
    if rank == 0:
        print(f"6: Took {e-s:.4f}s", flush=True)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if len(sys.argv) == 3:
        s = time.time()
        mesh = get_mesh(sys.argv[1])
        e = time.time()
        if rank == 0:
            print(f"1: Took {e-s:.4f}s", flush=True)

        if sys.argv[2] == "ic":
            solve(mesh, fname=sys.argv[1], cond_type="ic")
        elif sys.argv[2] == "bc":
            solve(mesh, fname=sys.argv[1], cond_type="bc")
        else:
            print("Wrong condition.")
    else:
        print("Must have filename.")
