#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

using namespace dealii;

class CSFSolver {
public:
    CSFSolver();
    void run();

    // Parameters
    double time_step, final_time;
    unsigned int refinements;
    unsigned int output_period;
    double grad_epsilon; 
    Function<2> *initial_condition;
    Function<2> *boundary_function;

private:
    void make_grid();
    void setup_system();
    void assemble_relaxation_step();
    // Computes the matices \int phi_i phi_j / (|grad u_l^{n + 1}| + |grad u^n|)
    // and \int grad phi_i . grad phi_j / (|grad u_l^{n + 1}| + |grad u^n|
    void compute_system_matrices(const Vector<double> &last_relax_step,
                                 const Vector<double> &previous_time_step,
                                 SparseMatrix<double> &mass_output,
                                 SparseMatrix<double> &stiffness_output);
    void solve_relaxation_step();
    void write_output(const unsigned int timestep_number);

    // Grid / FEM Components
    Triangulation<2>    triangulation;
    FE_Q<2>             fe;
    DoFHandler<2>       dof_handler;

    // Discrete solution components
    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    relax_system_matrix;
    Vector<double>          relax_rhs;
    Vector<double>          last_relax_rhs; // Keep this for residual computation

    // Dynamic variables
    double          current_time;
    Vector<double>  current_solution, last_solution;
};

CSFSolver::CSFSolver() :
    time_step (0.01),
    final_time (1.0),
    refinements (4),
    output_period (10),
    grad_epsilon (1e-12),
    // Internal
    fe (1),
    dof_handler (triangulation),
    current_time (0.0) {}

void CSFSolver::make_grid() {
    //GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    GridGenerator::hyper_ball(triangulation);
    static SphericalManifold<2> manifold_description;
    triangulation.set_manifold(0, manifold_description);

    for(auto cell : triangulation.active_cell_iterators()) {
        if(cell->at_boundary()) {
            cell->set_all_manifold_ids(0);
        }
    }

    triangulation.refine_global(refinements);

    //std::ofstream mesh_output("output/mesh.vtk");
    //GridOut().write_vtk(triangulation, mesh_output);
    
    std::cout << "Made grid. Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

void CSFSolver::setup_system() {
    dof_handler.distribute_dofs(fe);

    const unsigned int n_dofs = dof_handler.n_dofs();

    std::cout << "Setting up system. Number of dofs: " << n_dofs << std::endl;

    DynamicSparsityPattern dsp(n_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    relax_system_matrix.reinit(sparsity_pattern);

    relax_rhs.reinit(n_dofs);
    last_relax_rhs.reinit(n_dofs);
    current_solution.reinit(n_dofs);
    last_solution.reinit(n_dofs);
}

void CSFSolver::assemble_relaxation_step() {
    SparseMatrix<double> mass_matrix(sparsity_pattern);
    compute_system_matrices(current_solution, last_solution, 
                            mass_matrix, relax_system_matrix);

    // std::cout << "This is the stiffness matrix:" << std::endl;
    // relax_system_matrix.print(std::cout, 3, false);
    // Check against actual stiffness matrix
    // SparseMatrix<double> actual_stiff_mat(sparsity_pattern);
    // MatrixCreator::create_laplace_matrix(dof_handler, QGauss<2>(3), actual_stiff_mat);
    // actual_stiff_mat.print(std::cout, 3, false);

    // The system matrix is 1/k M + K
    relax_system_matrix.add(1.0/time_step, mass_matrix);

    // The right hand side is 1/k M u_n
    last_relax_rhs = relax_rhs; // Save the old rhs for residual computation
    mass_matrix.vmult(relax_rhs, last_solution);
    relax_rhs *= 1.0/time_step;
    //mass_matrix.print(std::cout, 3, false);

    // Check against actual mass matrix
    // SparseMatrix<double> actual_mass_mat(sparsity_pattern);
    // MatrixCreator::create_mass_matrix(dof_handler, QGauss<2>(3), actual_mass_mat);
    // actual_mass_mat.print(std::cout, 3, false);

    // Apply zero bcs
    std::map<types::global_dof_index, double> boundary_values;
    boundary_function->set_time(current_time);
    VectorTools::interpolate_boundary_values(dof_handler, 0, *boundary_function, boundary_values);
    MatrixTools::apply_boundary_values(boundary_values, relax_system_matrix, current_solution, relax_rhs);
}

void CSFSolver::compute_system_matrices(const Vector<double> &last_relax_step,
                                        const Vector<double> &previous_time_step,
                                        SparseMatrix<double> &mass_output,
                                        SparseMatrix<double> &stiffness_output) {
    mass_output = 0;
    stiffness_output = 0;
    const QGauss<2> quadrature_formula(3);
    FEValues<2> fe_values(fe, quadrature_formula,
        update_values | update_gradients | update_JxW_values | update_quadrature_points);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> local_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_stiffness_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<1, 2, double>> last_relax_grads(n_q_points);
    std::vector<Tensor<1, 2, double>> previous_step_grads(n_q_points);

    for(auto cell : dof_handler.active_cell_iterators()) {
        local_mass_matrix = 0;
        local_stiffness_matrix = 0;
        fe_values.reinit(cell);
        fe_values.get_function_gradients(last_relax_step, last_relax_grads);
        fe_values.get_function_gradients(previous_time_step, previous_step_grads);

        for(unsigned int q_pt_idx = 0; q_pt_idx < n_q_points; q_pt_idx++) {
            for(unsigned int i = 0; i < dofs_per_cell; i++) {
                for(unsigned int j = 0; j < dofs_per_cell; j++) {
                    // Compute the shared denominator - normalizing to avoid infinities
                    double grad_sum = last_relax_grads[q_pt_idx].norm() + previous_step_grads[q_pt_idx].norm();
                    double grad_sum_inv = 1.0 / MAX(grad_sum, grad_epsilon);
                    // Compute the mass component
                    local_mass_matrix(i, j) += (fe_values.shape_value(i, q_pt_idx) * fe_values.shape_value(j, q_pt_idx)  * grad_sum_inv * fe_values.JxW(q_pt_idx));
                    // Compute the stiffness component
                    local_stiffness_matrix(i, j) += (fe_values.shape_grad(i, q_pt_idx) * fe_values.shape_grad(j, q_pt_idx) * grad_sum_inv * fe_values.JxW(q_pt_idx));
                }
            }
        }

        // Copy the local matrices into the global output
        cell->get_dof_indices(local_dof_indices);
        for(unsigned int i = 0; i < dofs_per_cell; i++) {
            for(unsigned int j = 0; j < dofs_per_cell; j++) {
                mass_output.add(local_dof_indices[i], local_dof_indices[j], local_mass_matrix(i, j));
                stiffness_output.add(local_dof_indices[i], local_dof_indices[j], local_stiffness_matrix(i, j));
            }
        }
    }
}

void CSFSolver::solve_relaxation_step() {
    SolverControl solver_control(10000, 1e-12);
    SolverCG<> solver(solver_control);
    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(relax_system_matrix, 1.2);
    solver.solve(relax_system_matrix, current_solution, relax_rhs, preconditioner);
}

void CSFSolver::write_output(const unsigned int timestep_number) {  
    // std::ofstream out("output/solution-" + std::to_string(timestep_number) + ".csv");

    // for(auto cell : dof_handler.active_cell_iterators()) {
    //     for(unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; v++) {
    //         Point<2> pt = cell->vertex(v);
    //         out << pt(0) << ", " << pt(1) << ", " << current_solution(cell->vertex_dof_index(v, 0)) << std::endl;
    //     }
    // }

    DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(current_solution, "u");
    data_out.build_patches();

    std::ofstream out("output/solution-5-" + std::to_string(timestep_number) + ".vtk");
    data_out.write_vtk(out);
}

void CSFSolver::run() {
    // Setup
    make_grid();
    setup_system();

    // Initial condition
    ConstraintMatrix constraints;
    constraints.close();
    VectorTools::project(dof_handler, constraints, QGauss<2>(3),
        *initial_condition, current_solution);
    last_solution = current_solution;

    // Prime the right hand side for residual computation
    SparseMatrix<double> mass_matrix(sparsity_pattern);
    SparseMatrix<double> stiffness_matrix(sparsity_pattern); // Unused, but need to pass in something
    compute_system_matrices(current_solution, last_solution, 
                            mass_matrix, stiffness_matrix);
    mass_matrix.vmult(relax_rhs, last_solution);
    relax_rhs *= 1.0/time_step;

    write_output(0);

    // Begin time-stepping
    unsigned int total_steps = ceil(final_time / time_step);
    for(unsigned int step_idx = 1; step_idx <= total_steps; step_idx++) {
        current_time += time_step;
        std::cout << "=== (" << step_idx << ") Advancing time to " << current_time << " ===" << std::endl;

        assemble_relaxation_step();

        Vector<double> residual_vector(dof_handler.n_dofs());
        double residual = 0.0;
        unsigned int relaxation_steps = 0;
        do {
            if(relaxation_steps > 50) {
                std::cout << "Relaxation not converging. Abort." << std::endl;
                abort();
            }
            solve_relaxation_step();

            assemble_relaxation_step();
            //std::cout << "Norm of solution vector: " << current_solution.l2_norm() << std::endl;

            residual_vector = last_relax_rhs;
            residual_vector -= relax_rhs;
            residual = residual_vector.l2_norm();

            //std::cout << "Finished relaxation step. Residual: " << residual << std::endl;
            relaxation_steps++;
        } while(residual > 1e-8);

        std::cout << "Relaxation converged in " << relaxation_steps << " steps." << std::endl;

        last_solution = current_solution;

        if(step_idx % output_period == 0) {
            write_output(step_idx);
        }
    }
}

class TestInitialConditionQuadratic : public Function<2> {
public:
    TestInitialConditionQuadratic() : Function<2, double>() {}
    virtual double value(const Point<2> &pt, const unsigned int component = 0) const {
        const double x = pt(0);
        const double y = pt(1);
        return x * (1 - x) * y * (1 - y);
    }
};

// An exact radial solution to the level set equation, with dirichlet boundary conditions at r = 1
class TestInitialConditionExact : public Function<2> {
public:
    TestInitialConditionExact() : Function<2, double>() {}
    virtual double value(const Point<2> &pt, const unsigned int component = 0) const {
        const double x = pt(0);
        const double y = pt(1);
        const double r2 = std::pow(x, 2.0) + std::pow(y, 2.0);
        return std::exp(-r2/2.0) - std::exp(-0.5);
    }
};

class TestBoundaryFnExact : public Function<2> {
public:
    TestBoundaryFnExact() : Function<2, double>() {}
    virtual double value(const Point<2> &pt, const unsigned int component = 0) const {
        const double x = pt(0);
        const double y = pt(1);
        const double t = get_time();
        const double r2 = std::pow(x, 2.0) + std::pow(y, 2.0);
        return std::exp(-t-r2/2.0) - std::exp(-0.5);
    }
};

int main() {
    // First test, does zero stay zero? - yes, this works
    deallog.depth_console(0);

    CSFSolver solver;
    solver.initial_condition = new TestInitialConditionExact();
    solver.boundary_function = new TestBoundaryFnExact();
    solver.final_time = 0.125;
    solver.time_step = solver.final_time / 10.0;
    solver.output_period = 5;
    solver.refinements = 5;
    //solver.refinements = 1;
    //solver.final_time = 0.02;
    solver.run();
}
