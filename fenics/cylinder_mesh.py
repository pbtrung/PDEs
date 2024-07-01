import sys

import gmsh
from dolfinx.io import gmshio, XDMFFile
from mpi4py import MPI

from vars import *


def gen_cylinder_gmsh(fname_noext, mesh_size_max):
    gmsh.initialize()

    cy1 = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, l, r)
    cy2 = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, l, rtop)
    cy3 = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, l, r)
    gmsh.model.occ.cut([(3, cy1)], [(3, cy2)], removeObject=True, removeTool=False)

    # p100 = gmsh.model.occ.addPoint(0, 0, l)
    # p101 = gmsh.model.occ.addPoint(rtop, 0, l)
    # p102 = gmsh.model.occ.addPoint(0, rtop, l)
    # p103 = gmsh.model.occ.addPoint(-rtop, 0, l)
    # p104 = gmsh.model.occ.addPoint(0, -rtop, l)

    # c201 = gmsh.model.occ.addCircleArc(p101, p100, p102)
    # c202 = gmsh.model.occ.addCircleArc(p102, p100, p103)
    # c203 = gmsh.model.occ.addCircleArc(p103, p100, p104)
    # c204 = gmsh.model.occ.addCircleArc(p104, p100, p101)

    # curve = gmsh.model.occ.addCurveLoop([c201, c202, c203, c204])
    # plane = gmsh.model.occ.addPlaneSurface([curve])
    # ring = gmsh.model.occ.addPlaneSurface([2, plane])

    gmsh.model.occ.synchronize()

    # smallest 0.005
    # small    0.01
    # med      0.03
    # big      0.05
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    gmsh.option.setNumber("General.NumThreads", 50)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    # gmsh.model.addPhysicalGroup(dim2, [plane], tag=1)
    # gmsh.model.setPhysicalName(dim=dim2, tag=1, name="TopBoundary")
    # gmsh.model.addPhysicalGroup(dim2, [ring], tag=2)
    # gmsh.model.setPhysicalName(dim=dim2, tag=2, name="TopRing")
    # gmsh.model.addPhysicalGroup(dim2, [1], tag=3)
    # gmsh.model.setPhysicalName(dim=dim2, tag=3, name="Side")
    # gmsh.model.addPhysicalGroup(dim2, [3], tag=4)
    # gmsh.model.setPhysicalName(dim=dim2, tag=4, name="Bottom")

    # boundary_entities = gmsh.model.getEntities(2)
    # other_boundary_entities = []
    # for entity in boundary_entities:
    #     other_boundary_entities.append(entity[1])
    # gmsh.model.addPhysicalGroup(2, other_boundary_entities, tag=1)
    # gmsh.model.setPhysicalName(dim=dim2, tag=1, name="boundary")

    gmsh.model.addPhysicalGroup(dim3, [cy3], tag=1)
    gmsh.model.setPhysicalName(dim=dim3, tag=1, name="cylinder")

    gmsh.model.addPhysicalGroup(dim2, [6, 9], tag=1)
    gmsh.model.setPhysicalName(dim=dim2, tag=1, name="bottom")
    gmsh.model.addPhysicalGroup(dim2, [5], tag=2)
    gmsh.model.setPhysicalName(dim=dim2, tag=2, name="top_boundary")
    gmsh.model.addPhysicalGroup(dim2, [8], tag=3)
    gmsh.model.setPhysicalName(dim=dim2, tag=3, name="ring")
    gmsh.model.addPhysicalGroup(dim2, [7], tag=4)
    gmsh.model.setPhysicalName(dim=dim2, tag=4, name="surface")

    gmsh.model.mesh.generate(dim3)
    gmsh.write(fname_noext + ".msh")
    gmsh.finalize()


def gen_cylinder_xdmf(fname_noext):
    mesh, cell_markers, facet_markers = gmshio.read_from_msh(
        fname_noext + ".msh", MPI.COMM_WORLD, rank=0, gdim=dim3
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
