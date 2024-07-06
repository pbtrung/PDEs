#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

class ConvectionDiffusionOperator : public TimeDependentOperator {
  private:
    ParFiniteElementSpace &fespace;
    ParBilinearForm *M;
    ParBilinearForm *K;
    HypreParMatrix Mmat, Kmat;
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
          c0(c0), cg(fespace.GetComm()), ess_tdof_list(ess_tdof_list) {
        int skip_zeros = 0;

        M = new ParBilinearForm(&fespace);
        K = new ParBilinearForm(&fespace);
        M->AddDomainIntegrator(new MassIntegrator());
        K->AddDomainIntegrator(new ConvectionIntegrator(vCoeff));
        K->AddDomainIntegrator(new DiffusionIntegrator(dCoeff));

        M->Assemble(skip_zeros);
        M->Finalize(skip_zeros);
        K->Assemble(skip_zeros);
        K->Finalize(skip_zeros);

        M->FormSystemMatrix(ess_tdof_list, Mmat);
        Array<int> empty;
        K->FormSystemMatrix(empty, Kmat);

        cg.iterative_mode = false;
        cg.SetRelTol(1e-12);
        cg.SetAbsTol(1e-12);
        cg.SetMaxIter(1000);
        cg.SetPrintLevel(1);
        prec.SetType(HypreSmoother::Jacobi);
        cg.SetPreconditioner(prec);
    }

    virtual void ImplicitSolve(const double dt, const Vector &x, Vector &y) {
        HypreParMatrix A(Mmat);
        A.Add(dt, Kmat);
        cg.SetOperator(A);
        Vector B(x.Size());
        Mmat.Mult(x, B);
        cg.Mult(B, y);
        y.SetSubVector(ess_tdof_list, c0);
    }

    virtual ~ConvectionDiffusionOperator() {
        delete M;
        delete K;
    }
};

int main(int argc, char *argv[]) {
    Mpi::Init();
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    const char *mesh_prefix = "cylinder.mesh.";
    int order = 1;
    const char *device_config = "cpu";

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_prefix, "-p", "--prefix",
                   "Prefix for parallel mesh files.");

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
    ifstream mesh_ifs(MakeParFilename(mesh_prefix, myid));
    ParMesh pmesh(MPI_COMM_WORLD, mesh_ifs, false);
    int dim = pmesh.Dimension();

    FiniteElementCollection *fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace fespace(&pmesh, fec);
    HYPRE_BigInt size = fespace.GlobalTrueVSize();
    if (myid == 0) {
        cout << "Number of finite element unknowns: " << size << endl;
        cout << "1: " << toc() << endl;
    }

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
    double dt = 0.01;
    double t_final = 2.0;
    int step = 0;

    Vector u;
    c.GetTrueDofs(u);

    ADIOS2DataCollection dataCol(MPI_COMM_WORLD, "results.bp", &pmesh);
    dataCol.SetLevelsOfDetail(1);
    dataCol.RegisterField("concentration", &c);
    dataCol.SetCycle(step);
    dataCol.SetTime(t);
    dataCol.Save();

    while (t < t_final) {
        t += dt;

        tic();
        oper.ImplicitSolve(dt, u, u);
        c.SetFromTrueDofs(u);
        step++;
        if (myid == 0) {
            cout << "2: " << toc() << endl;
            cout << "Step " << step << ", Time " << t
                 << ", Norm of solution: " << c.Norml2() << endl;
        }

        dataCol.SetCycle(step);
        dataCol.SetTime(t);
        dataCol.Save();
    }

    delete fec;
    return 0;
}