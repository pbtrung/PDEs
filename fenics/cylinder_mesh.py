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
    ci = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.addPhysicalGroup(gdim, [cy], 1)
    gmsh.model.addPhysicalGroup(gdim, [ci], 2)
    gmsh.model.setPhysicalName(gdim, 1, "m1")
    gmsh.model.setPhysicalName(gdim, 2, "m2")
    gmsh.model.mesh.generate(gdim)

    gmsh.write(fname + ".msh")
    # gmsh.fltk.run()
    gmsh.finalize()


def gen_cylinder_xdmf(fname, fname_phys, physical_group):
    if not physical_group:
        mesh, cell_markers, facet_markers = gmshio.read_from_msh(
            fname + ".msh", MPI.COMM_WORLD, 0, gdim=gdim
        )

        with XDMFFile(mesh.comm, fname + ".xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            print("done!")
    else:
        msh, ct, ft = gmshio.read_from_msh(fname_phys + ".msh", MPI.COMM_WORLD, 0, gdim=gdim)
        msh.name = "mesh"
        ct.name = f"{msh.name}_cells"
        ft.name = f"{msh.name}_facets"
        print(ft.find(1))
        print(ct.find(2))
        with XDMFFile(msh.comm, fname_phys + ".xdmf", "w") as file:
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
    if len(sys.argv) == 4:
        if sys.argv[3] != "phys":
            gen_cylinder_gmsh(sys.argv[1])
            gen_cylinder_xdmf(sys.argv[1], sys.argv[2], False)
        else:
            gen_cylinder_gmsh(sys.argv[1])
            gen_cylinder_xdmf(sys.argv[1], sys.argv[2], True)
    else:
        print("Must have filename.")
