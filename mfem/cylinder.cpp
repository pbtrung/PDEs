#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Custom TimeDependentOperator for the PDE
class ConvectionDiffusionOperator : public TimeDependentOperator {
  protected:
    BilinearForm *m, *k, *adv;
    SparseMatrix M, K, Adv;
    Array<int> ess_tdof_list;

  public:
    ConvectionDiffusionOperator(FiniteElementSpace &fespace, Vector &velocity,
                                double diffusion_coeff)
        : TimeDependentOperator(fespace.GetTrueVSize(), 0.0) {
        m = new BilinearForm(&fespace);
        k = new BilinearForm(&fespace);
        adv = new BilinearForm(&fespace);

        m->AddDomainIntegrator(new MassIntegrator());
        k->AddDomainIntegrator(
            new DiffusionIntegrator(ConstantCoefficient(diffusion_coeff)));
        adv->AddDomainIntegrator(new ConvectionIntegrator(
            VectorConstantCoefficient(velocity), -1.0));

        m->Assemble();
        m->Finalize();
        k->Assemble();
        k->Finalize();
        adv->Assemble();
        adv->Finalize();

        M = m->SpMat();
        K = k->SpMat();
        Adv = adv->SpMat();

        Adv.Add(1.0, K); // Adv becomes Adv + K

        Array<int> ess_bdr;
        ess_bdr.SetSize(fespace.GetMesh()->bdr_attributes.Max());
        ess_bdr = 0;
        ess_bdr[1] = 1; // Assuming attribute 1 is the top circle

        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    virtual void Mult(const Vector &u, Vector &du_dt) const {
        Vector z(u.Size());
        Adv.Mult(u, z);
        M.Mult(z, du_dt);
    }

    virtual void ImplicitSolve(const double dt, const Vector &u,
                               Vector &du_dt) {
        SparseMatrix A;
        A.Add(1.0, Adv);
        A.Add(1.0 / dt, M);

        GSSmoother M_inv(M);
        PCG(A, M_inv, du_dt, du_dt, 0, 200, 1e-12, 0.0);
    }

    virtual ~ConvectionDiffusionOperator() {
        delete m;
        delete k;
        delete adv;
    }
};

int main(int argc, char *argv[]) {
    // Initialize the MFEM library.
    const char *mesh_file = "cylinder.mesh";
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // Define a finite element space on the mesh.
    int order = 1;
    H1_FECollection fec(order, dim);
    FiniteElementSpace fespace(mesh, &fec);

    // Define the solution vector 'c' as a grid function.
    GridFunction c(&fespace);
    c = 0.0; // Initialize with zero concentration

    // Define the boundary condition.
    Array<int> top_boundary(mesh->bdr_attributes.Max());
    top_boundary = 0;
    top_boundary[1] = 1; // Assuming attribute 1 corresponds to the top circle
    ConstantCoefficient one(1.0);
    c.ProjectBdrCoefficient(one, top_boundary);

    // Define the velocity vector v.
    Vector v(dim);
    v = 0.0;
    v[2] = -0.25;

    // Define the diffusion coefficient.
    double diffusion_coeff = 0.02;

    // Define the time-dependent operator.
    ConvectionDiffusionOperator oper(fespace, v, diffusion_coeff);

    // Define the initial condition.
    c = 0.0; // Assuming initial concentration is zero

    // Set up the time integration.
    double t = 0.0;
    double t_final = 1.0;
    double dt = 0.01;

    // RK4 time integrator
    RK4Solver ode_solver;
    ode_solver.Init(oper);

    // Time integration loop.
    while (t < t_final) {
        ode_solver.Step(c, t, dt);
    }

    // Output the result.
    ofstream sol_ofs("solution.gf");
    sol_ofs.precision(8);
    c.Save(sol_ofs);

    delete mesh;

    return 0;
}
