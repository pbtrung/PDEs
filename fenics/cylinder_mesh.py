import sys

import gmsh
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI

from vars import *


def gen_cylinder_gmsh(fname_noext, mesh_size_max):
    gmsh.initialize()

    cy = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, l, r)

    p100 = gmsh.model.occ.addPoint(0, 0, l)
    p101 = gmsh.model.occ.addPoint(rtop, 0, l)
    p102 = gmsh.model.occ.addPoint(0, rtop, l)
    p103 = gmsh.model.occ.addPoint(-rtop, 0, l)
    p104 = gmsh.model.occ.addPoint(0, -rtop, l)

    c201 = gmsh.model.occ.addCircleArc(p101, p100, p102)
    # Circle(201) = {101, 100, 102};
    # Circle(202) = {102, 100, 103};
    # Circle(203) = {103, 100, 104};
    # Circle(204) = {104, 100, 101};
    gmsh.model.occ.synchronize()

    # smallest 0.005
    # small    0.01
    # med      0.03
    # big      0.05
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.model.addPhysicalGroup(dim3, [cy], tag=1)
    gmsh.model.setPhysicalName(dim=dim3, tag=1, name="cylinder")
    gmsh.model.mesh.generate(dim3)

    gmsh.write(fname_noext + ".msh")
    gmsh.finalize()


def gen_cylinder_xdmf(fname_noext):
    mesh, cell_markers, facet_markers = gmshio.read_from_msh(
        fname_noext + ".msh", MPI.COMM_WORLD, 0, gdim=dim3
    )
    mesh.name = "mesh"
    cell_markers.name = f"{mesh.name}_cells"
    facet_markers.name = f"{mesh.name}_facets"
    with XDMFFile(mesh.comm, fname_noext + ".xdmf", "w") as file:
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
        print("done!")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        fname_noext = sys.argv[1]
        mesh_size_max = float(sys.argv[2])
        gen_cylinder_gmsh(fname_noext, mesh_size_max)
        gen_cylinder_xdmf(fname_noext)
    else:
        print("Must have filename.")
