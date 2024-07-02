#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

int main(int argc, char *argv[]) {
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    const char *mesh_file = "cylinder.msh";
    int order = 1;
    bool pa = false;
    bool fa = false;
    const char *device_config = "cpu";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");

    args.Parse();
    if (!args.Good()) {
        if (myid == 0) {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0) {
        args.PrintOptions(cout);
    }

    Device device(device_config);
    if (myid == 0) {
        device.Print();
    }

    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    FiniteElementCollection *fec = new H1_FECollection(order, dim);

    ParFiniteElementSpace fespace(&pmesh, fec);
    HYPRE_BigInt size = fespace.GlobalTrueVSize();
    if (myid == 0) {
        cout << "Number of finite element unknowns: " << size << endl;
    }

    delete fec;
    return 0;
}