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
    HypreParMatrix *Mmat, *Kmat;
    Array<int> ess_tdof_list;
    double c0 = 1.0;

    CGSolver cg;
    HypreSmoother prec;

  public:
    ConvectionDiffusionOperator(ParFiniteElementSpace &fespace,
                                VectorCoefficient &vCoeff,
                                ConstantCoefficient &dCoeff,
                                Array<int> &ess_tdof_list, double c0)
        : TimeDependentOperator(fespace.GetTrueVSize(), 0.0), fespace(fespace),
          vCoeff(&vCoeff), dCoeff(&dCoeff), ess_tdof_list(ess_tdof_list),
          c0(c0) {
        convInteg = new ConvectionIntegrator(vCoeff);
        diffInteg = new DiffusionIntegrator(dCoeff);
        M = new ParBilinearForm(&fespace);
        K = new ParBilinearForm(&fespace);
        M->AddDomainIntegrator(new MassIntegrator());
        K->AddDomainIntegrator(convInteg);
        K->AddDomainIntegrator(diffInteg);
        M->Assemble(0);
        M->Finalize();
        K->Assemble(0);
        K->Finalize();

        Mmat = M->ParallelAssemble();
        // HypreParMatrix *tmp = Mmat->EliminateRowsCols(ess_tdof_list);
        // delete tmp;
        Kmat = K->ParallelAssemble();
        // tmp = Kmat->EliminateRowsCols(ess_tdof_list);
        // delete tmp;

        cg.iterative_mode = false;
        cg.SetRelTol(1e-12);
        cg.SetAbsTol(0.0);
        cg.SetMaxIter(1000);
        cg.SetPrintLevel(1);
        prec.SetType(HypreSmoother::Jacobi);
        cg.SetPreconditioner(prec);
    }

    virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y) {
        HypreParMatrix A(*Mmat);
        A.Add(dt, *Kmat);
        cg.SetOperator(A);
        Vector B(x.Size());
        cout << "Mmat row: " << Mmat->NumRows() << endl;
        cout << "Mmat col: " << Mmat->NumCols() << endl;
        cout << "B: " << B.Size() << endl;
        cout << "x: " << x.Size() << endl;
        Mmat->Mult(x, B);
        cg.Mult(B, y);
        y.SetSubVector(ess_tdof_list, c0);
    }

    virtual ~ConvectionDiffusionOperator() {
        delete convInteg;
        delete diffInteg;
        delete M;
        delete Mmat;
        delete Kmat;
        delete K;
    }
};

int main(int argc, char *argv[]) {
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    const char *mesh_file = "cylinder.mesh";
    int order = 1;
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

    int top_boundary_attr = 2;
    ess_bdr = 0;
    ess_bdr[top_boundary_attr - 1] = 1;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    double c0 = 1.0;
    ConstantCoefficient cc0(c0);
    c.ProjectBdrCoefficient(cc0, ess_bdr);

    Vector v(3);
    v = 0.0;
    v(2) = -0.25;
    double d = 0.02;
    VectorConstantCoefficient vCoeff(v);
    ConstantCoefficient dCoeff(d);

    ConvectionDiffusionOperator oper(fespace, vCoeff, dCoeff, ess_tdof_list,
                                     c0);

    double t = 0.0;
    double t_final = 1.0;
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

        t += dt;
        tic();
        oper.ImplicitSolve(dt, c, c);
        cout << "2: " << toc() << endl;
        step++;

        if (myid == 0) {
            cout << "Step " << step << ", Time " << t
                 << ", Norm of solution: " << c.Norml2() << endl;
        }
        if (step == 10) {
            break;
        }
    }

    delete fec;
    return 0;
}
