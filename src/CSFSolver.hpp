#ifndef CSF_SOLVER_INCLUDE

#define CSF_SOLVER_INCLUDE

#define DEFAULT_TIME_STEP 0.01
#define DEFAULT_FINAL_TIME 1.0
#define DEFAULT_GRAD_EPSILON 1e-12
#define DEFAULT_RESIDUAL_TOLERANCE 1e-8
#define DEFAULT_MAX_RELAX_STEPS 500

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>


using namespace dealii;

/**
CSFSolver is an abstract class that handles FEM solutions to 2D curve shortening flow.

To implement a solver for a particular problem, the following overrides are required:
void make_grid() -- construct a mesh and store in 'triangulation'
void write_output(const unsigned int timestep_number) -- called at each time step, responsible for deciding when to write output, and doing write

The initial level set field is specified by setting 'initial_condition'.
The dirichlet boundary conditions (allowed to be time-dependent) are specified by setting 'boundary_function'.

The 'void run()' method calls make_grid, then constructs matrices and begins time stepping. Asserted preconditions:
- initial_condition or boundary_function are NULL
Exceptions are thrown in the following cases:
- relaxation fails to converge to within relaxation_residual_tolerance after max_relaxation_steps
**/

DeclException1(RelaxationConvergenceFailure, double, << "Residual = " << arg1);

class CSFSolver {
public:
    CSFSolver();
    void run();

    // Parameters
    double time_step, final_time;
    double grad_epsilon;
    double relaxation_residual_tolerance;
    unsigned int max_relaxation_steps;
    bool time_dep_boundary_conditions;
    bool use_scheduled_relaxation;
    Function<2> *initial_condition;
    Function<2> *boundary_function;

    // Extension points
    virtual void make_grid() = 0;
    virtual void write_output(const unsigned int timestep_number) = 0;

private:
    void setup_system();
    void assemble_relaxation_step();
    // Computes the matices \int phi_i phi_j / (|grad u_l^{n + 1}| + |grad u^n|)
    // and \int grad phi_i . grad phi_j / (|grad u_l^{n + 1}| + |grad u^n|
    void compute_system_matrices(const Vector<double> &last_relax_step,
                                 const Vector<double> &previous_time_step,
                                 SparseMatrix<double> &mass_output,
                                 SparseMatrix<double> &stiffness_output);
    void solve_relaxation_step();

    // Grid / FEM Components
    FE_Q<2>             fe;

    // Discrete solution components
    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    relax_system_matrix;
    Vector<double>          relax_rhs;
    Vector<double>          last_relax_rhs; // Keep this for residual computation

    // Stored boundary values
    std::map<dealii::types::global_dof_index, double> boundary_values;

protected:
    // Grid / FEM Components
    Triangulation<2> triangulation;
    DoFHandler<2>    dof_handler;

    // Dynamic variables
    double          current_time;
    Vector<double>  current_solution, last_solution;
};

#endif
