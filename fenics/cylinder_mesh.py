import gmsh
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI

from vars import *


def gen_cylinder_gmsh():
    gmsh.initialize()

    gmsh.model.add("cylinder")
    # x, y, z, dx, dy, dz, r
    cy = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, l, r)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
    gmsh.model.addPhysicalGroup(gdim, [cy], 1)
    gmsh.model.mesh.generate(gdim)

    gmsh.write(fname + ".msh")
    # gmsh.fltk.run()
    gmsh.finalize()


def gen_cylinder_xdmf():
    mesh, cell_markers, facet_markers = gmshio.read_from_msh(
        fname + ".msh", MPI.COMM_WORLD, 0, gdim=gdim
    )

    with XDMFFile(mesh.comm, fname + ".xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        print("done!")


if __name__ == "__main__":
    gen_cylinder_gmsh()
    gen_cylinder_xdmf()
