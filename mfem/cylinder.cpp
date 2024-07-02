#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ConvectionDiffusionOperator : public TimeDependentOperator {
  private:
    ParFiniteElementSpace &fespace;
    VectorCoefficient *vCoeff;
    ConstantCoefficient *dCoeff;
    ConvectionIntegrator *convInteg;
    DiffusionIntegrator *diffInteg;
    ParBilinearForm *M;
    ParBilinearForm *K;

  public:
    ConvectionDiffusionOperator(ParFiniteElementSpace &fespace,
                                VectorCoefficient &vCoeff,
                                ConstantCoefficient &dCoeff)
        : TimeDependentOperator(fespace.GetTrueVSize(), 0.0), fespace(fespace),
          vCoeff(&vCoeff), dCoeff(&dCoeff) {
        convInteg = new ConvectionIntegrator(vCoeff);
        diffInteg = new DiffusionIntegrator(dCoeff);
        M = new ParBilinearForm(&fespace);
        K = new ParBilinearForm(&fespace);
        M->AddDomainIntegrator(new MassIntegrator());
        K->AddDomainIntegrator(convInteg);
        K->AddDomainIntegrator(diffInteg);
        M->Assemble();
        M->Finalize();
        K->Assemble();
        K->Finalize();
    }

    virtual void Mult(const Vector &x, Vector &y) const { K->Mult(x, y); }

    virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y) {
        HypreParMatrix A(M->SpMat());
        A.Add(dt, Kmat->SpMat());

        CGSolver cg;
        cg.SetOperator(A);
        cg.SetRelTol(1e-12);
        cg.SetMaxIter(1000);
        cg.SetPrintLevel(0);
        Vector B(Mmat.Height());
        Mmat.Mult(x, B);
        cg.Mult(B, y);
    }

    virtual ~ConvectionDiffusionOperator() {
        delete convInteg;
        delete diffInteg;
        delete M;
        delete K;
    }
};

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

    cout << "Boundary attributes:" << endl;
    for (int i = 0; i < mesh.bdr_attributes.Size(); i++) {
        cout << "Boundary attribute " << i << ": " << mesh.bdr_attributes[i]
             << endl;
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
    int top_boundary_attr = 2;
    ess_bdr[top_boundary_attr - 1] = 1;

    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    ConstantCoefficient one(1.0);
    c.ProjectCoefficient(one, ess_tdof_list);

    Vector v(3);
    v = 0.0;
    v(2) = -0.25;
    double d = 0.02;
    VectorConstantCoefficient vCoeff(v);
    ConstantCoefficient dCoeff(d);

    ConvectionDiffusionOperator oper(fespace, vCoeff, dCoeff);
    BackwardEulerSolver ode_solver;
    ode_solver.Init(oper);

    double t = 0.0;
    double t_final = 5.0;
    double dt = 0.01;
    int step = 0;

    ParaViewDataCollection pd("cylinder", &pmesh);
    pd.SetPrefixPath("ParaView");
    pd.RegisterField("solution", &c);
    pd.SetLevelsOfDetail(order);
    pd.SetDataFormat(VTKFormat::BINARY);
    pd.SetHighOrderOutput(true);

    while (t < t_final) {
        pd.SetCycle(step);
        pd.SetTime(t);
        pd.Save();

        tic();
        ode_solver.Step(c, t, dt);
        cout << "2: " << toc() << endl;
        step++;
        if (step == 5) {
            break;
        }
    }

    delete fec;
    return 0;
}