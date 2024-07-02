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
    CGSolver cg;
    Array<int> ess_tdof_list;
    OperatorHandle Mmat, Kmat;
    // ParLinearForm *bform;
    // Vector *b = nullptr;
    // mutable Vector t1, t2;

  public:
    ConvectionDiffusionOperator(ParFiniteElementSpace &fespace,
                                VectorCoefficient &vCoeff,
                                ConstantCoefficient &dCoeff,
                                Array<int> ess_tdof_list)
        : TimeDependentOperator(fespace.GetTrueVSize(), 0.0), fespace(fespace),
          vCoeff(&vCoeff), dCoeff(&dCoeff), ess_tdof_list(ess_tdof_list) {
        convInteg = new ConvectionIntegrator(vCoeff);
        diffInteg = new DiffusionIntegrator(dCoeff);
        M = new ParBilinearForm(&fespace);
        K = new ParBilinearForm(&fespace);
        // bform = new ParLinearForm(&fespace);
        M->AddDomainIntegrator(new MassIntegrator());
        K->AddDomainIntegrator(convInteg);
        K->AddDomainIntegrator(diffInteg);
        M->Assemble(0);
        K->Assemble(0);

        Array<int> empty;
        K->FormSystemMatrix(empty, Kmat);
        M->FormSystemMatrix(ess_tdof_list, Mmat);
        // bform->Assemble();
        // b = bform->ParallelAssemble();

        // t1.SetSize(Mmat->Height());
        // t2.SetSize(Mmat->Height());

        // cg.SetOperator(*Mmat);
        // cg.SetRelTol(1e-12);
        // cg.SetAbsTol(0.0);
        // cg.SetMaxIter(1000);
        // cg.SetPrintLevel(0);
    }

    // virtual void Mult(const Vector &x, Vector &y) const { K->Mult(x, y); }
    // void Mult(const Vector &u, Vector &du_dt) const override {
    //     Kmat->Mult(u, t1);
    //     t1.Add(1.0, *b);
    //     cg.Mult(t1, du_dt);
    //     du_dt.SetSubVector(ess_tdof_list, 1.0);
    // }

    virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y) {
        int size = Mmat.Height();

        SparseMatrix A(Mmat->SpMat());
        A.Add(dt, Kmat->SpMat());

        cg.SetOperator(A);
        cg.SetRelTol(1e-12);
        cg.SetAbsTol(0.0);
        cg.SetMaxIter(1000);
        cg.SetPrintLevel(0);
        Vector B(size);
        Mmat.Mult(x, B);
        cg.Mult(B, y);

        y.SetSubVector(ess_tdof_list, 1.0);
    }

    virtual ~ConvectionDiffusionOperator() {
        delete convInteg;
        delete diffInteg;
        delete M;
        delete K;
        // delete bform;
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

    ConvectionDiffusionOperator oper(fespace, vCoeff, dCoeff, ess_tdof_list);
    BackwardEulerSolver ode_solver;
    // RK3SSPSolver ode_solver;
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

        // t += dt;
        tic();
        ode_solver.Step(c, t, dt);
        cout << "2: " << toc() << endl;
        step++;
        if (step == 10) {
            break;
        }
    }

    delete fec;
    return 0;
}