import sys

import gmsh
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI

from vars import *


def gen_cylinder_gmsh(fname):
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


def gen_cylinder_xdmf(fname, physical_group):
    if not physical_group:
        mesh, cell_markers, facet_markers = gmshio.read_from_msh(
            fname + ".msh", MPI.COMM_WORLD, 0, gdim=gdim
        )

        with XDMFFile(mesh.comm, fname + ".xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            print("done!")
    else:
        msh, ct, ft = gmshio.read_from_msh(fname + ".msh", MPI.COMM_WORLD, 0, gdim=gdim)
        msh.name = "mesh"
        ct.name = f"{msh.name}_cells"
        ft.name = f"{msh.name}_facets"
        with XDMFFile(msh.comm, fname + ".xdmf", "w") as file:
            msh.topology.create_connectivity(2, 3)
            file.write_mesh(msh)
            file.write_meshtags(
                ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
            )
            file.write_meshtags(
                ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
            )
            print("done!")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        if sys.argv[2] != "phys":
            gen_cylinder_gmsh(sys.argv[1])
            gen_cylinder_xdmf(sys.argv[1], False)
        else:
            gen_cylinder_xdmf(sys.argv[1], True)
    else:
        print("Must have filename.")
