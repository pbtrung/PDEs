#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ConvectionDiffusionOperator : public TimeDependentOperator {
  private:
    FiniteElementSpace &fespace;
    VectorCoefficient *vCoeff;
    ConstantCoefficient *dCoeff;
    ConvectionIntegrator *convInteg;
    DiffusionIntegrator *diffInteg;
    ParBilinearForm *M;
    ParBilinearForm *K;
    HypreParMatrix Mmat, Kmat;
    Array<int> ess_tdof_list;

    // ParLinearForm *bform;
    // Vector *b = nullptr;

    CGSolver cg;
    HypreSmoother cg_prec;

    // Auxiliary vectors
    // mutable Vector z;

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
        M->AddDomainIntegrator(new MassIntegrator());
        K->AddDomainIntegrator(convInteg);
        K->AddDomainIntegrator(diffInteg);
        M->Assemble(0);
        M->FormSystemMatrix(ess_tdof_list, Mmat);
        K->Assemble(0);
        K->FormSystemMatrix(ess_tdof_list, Kmat);

        // bform = new ParLinearForm(&fespace);
        // bform->Assemble();
        // b = bform->ParallelAssemble();

        cg.iterative_mode = false;
        cg.SetRelTol(1e-12);
        cg.SetAbsTol(0.0);
        cg.SetMaxIter(1000);
        cg.SetPrintLevel(0);
        cg_prec.SetType(HypreSmoother::Jacobi);
        cg.SetPreconditioner(cg_prec);

        // z.SetSize(Mmat.Height());
    }

    // void Mult(const Vector &u, Vector &du_dt) const override {
    //     Kmat.Mult(u, z);
    //     z.Neg();
    //     z.Add(1.0, *b);
    //     cg.Mult(z, du_dt);
    //     du_dt.SetSubVector(ess_tdof_list, 0.0);
    // }

    virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y) {
        HypreParMatrix A(Mmat);
        A.Add(dt, Kmat);

        cg.SetOperator(A);
        Vector B(Mmat.Height());
        Mmat.Mult(x, B);
        cg.Mult(B, y);
    }

    virtual ~ConvectionDiffusionOperator() {
        delete convInteg;
        delete diffInteg;
        delete M;
        // delete bform;
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

    ParGridFunction c_gf(&fespace);
    c_gf = 0.0;

    // Define the boundary condition
    Array<int> ess_tdof_list;
    Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
    ess_bdr = 0;
    int top_boundary_attr = 2;
    ess_bdr[top_boundary_attr - 1] = 1;

    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    ConstantCoefficient one(1.0);
    c_gf.ProjectBdrCoefficient(one, ess_bdr);
    Vector c;
    c_gf.GetTrueDofs(c);

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
    double t_final = 1.0;
    double dt = 0.01;
    int step = 0;

    ParaViewDataCollection pd("cylinder", &pmesh);
    pd.SetPrefixPath("ParaView");
    pd.RegisterField("solution", &c_gf);
    pd.SetLevelsOfDetail(order);
    pd.SetDataFormat(VTKFormat::BINARY);
    pd.SetHighOrderOutput(true);

    while (t < t_final) {
        pd.SetCycle(step);
        pd.SetTime(t);
        pd.Save();

        tic();
        ode_solver.Step(c, t, dt);
        c_gf.SetFromTrueDofs(c);
        cout << "2: " << toc() << endl;
        step++;
        if (step == 10) {
            break;
        }
    }

    delete fec;
    return 0;
}