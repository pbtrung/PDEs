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

    tic();
    Mesh mesh(mesh_file, 1, 1);
    int dim = mesh.Dimension();

    int top_boundary_attr = 0;
    cout << "Boundary attributes:" << endl;
    for (int i = 0; i < mesh.bdr_attributes.Size(); i++) {
        cout << "Boundary attribute " << i << ": " << mesh.bdr_attributes[i]
             << endl;
        if (mesh.bdr_attributes[i] == "top_boundary") {
            top_boundary_attr = i;
            cout << "top_boundary_attr " << top_boundary_attr << endl;
        }
    }

    ParMesh pmesh(MPI_COMM_WORLD, mesh);
    mesh.Clear();

    FiniteElementCollection *fec = new H1_FECollection(order, dim);

    ParFiniteElementSpace fespace(&pmesh, fec);
    HYPRE_BigInt size = fespace.GlobalTrueVSize();
    if (myid == 0) {
        cout << "Number of finite element unknowns: " << size << endl;
    }
    cout << "1: " << toc() << endl;

    ParGridFunction c(&fespace);
    c = 0.0;

    // Define the boundary condition
    Array<int> ess_tdof_list;
    Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[top_boundary_attr - 1] = 1;

    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // Apply the boundary condition
    Vector x;
    c.GetTrueDofs(x);
    for (int i = 0; i < ess_tdof_list.Size(); i++) {
        x[ess_tdof_list[i]] = 1.0;
    }
    c.SetFromTrueDofs(x);

    double t = 0.0;
    double t_final = 1.0;
    double dt = 0.01;
    int step = 0;

    ParaViewDataCollection pd("test", &pmesh);
    pd.SetPrefixPath("ParaView");
    pd.RegisterField("solution", &c);
    pd.SetLevelsOfDetail(order);
    pd.SetDataFormat(VTKFormat::BINARY);
    pd.SetHighOrderOutput(true);

    while (t < t_final) {
        tic();
        t += dt;
        c = t;
        pd.SetCycle(step);
        pd.SetTime(t);
        pd.Save();
        step++;
        cout << "2: " << toc() << endl;
        if (step == 10) {
            break;
        }
    }

    delete fec;
    return 0;
}