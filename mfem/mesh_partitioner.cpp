#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void recover_bdr_partitioning(const Mesh *mesh, const Array<int> &partitioning,
                              Array<int> &bdr_partitioning) {
    bdr_partitioning.SetSize(mesh->GetNBE());
    int info, e;
    for (int be = 0; be < mesh->GetNBE(); be++) {
        mesh->GetBdrElementAdjacentElement(be, e, info);
        bdr_partitioning[be] = partitioning[e];
    }
}

int main(int argc, char *argv[]) {
    const char *mesh_file = "cylinder.mesh";
    const char *mesh_prefix = "cylinder.mesh.";
    int np = 1;
    int part_method = 1;
    int precision = 14;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
    args.AddOption(&np, "-np", "--num-proc", "Number of processors.");
    args.AddOption(&mesh_prefix, "-p", "--prefix",
                   "Prefix for mesh partitioner.");

    args.Parse();
    if (!args.Good()) {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    Array<int> partitioning;
    Array<int> bdr_partitioning;
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    partitioning.SetSize(mesh->GetNE());
    partitioning = 0;
    bdr_partitioning.SetSize(mesh->GetNBE());
    bdr_partitioning = 0;

    int *part = mesh->GeneratePartitioning(np, part_method);
    partitioning = Array<int>(part, mesh->GetNE());
    delete[] part;
    recover_bdr_partitioning(mesh, partitioning, bdr_partitioning);

    MeshPartitioner partitioner(*mesh, np, partitioning);
    MeshPart mesh_part;

    for (int i = 0; i < np; i++) {
        partitioner.ExtractPart(i, mesh_part);
        ofstream omesh(MakeParFilename(mesh_prefix, i));
        mesh_part.Print(omesh);
    }
    cout << "New parallel mesh files: " << mesh_prefix << "<rank>" << endl;

    delete mesh;
    return 0;
}