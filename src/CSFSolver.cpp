#include "CSFSolver.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/logstream.h>

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

#include <cassert>
#include <fstream>

using namespace dealii;

CSFSolver::CSFSolver() :
    time_step (DEFAULT_TIME_STEP),
    final_time (DEFAULT_FINAL_TIME),
    grad_epsilon (DEFAULT_GRAD_EPSILON),
    relaxation_residual_tolerance (DEFAULT_RESIDUAL_TOLERANCE),
    max_relaxation_steps (DEFAULT_MAX_RELAX_STEPS),
    time_dep_boundary_conditions (true),
    // Internal
    setup_done (false),
    fe (1),
    dof_handler (triangulation),
    current_time (0.0) {}

void CSFSolver::setup_system() {
    make_grid();

    dof_handler.distribute_dofs(fe);

    const unsigned int n_dofs = dof_handler.n_dofs();

    deallog << "Setting up system. Number of dofs: " << n_dofs << std::endl;

    DynamicSparsityPattern dsp(n_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    relax_system_matrix.reinit(sparsity_pattern);

    relax_rhs.reinit(n_dofs);
    current_solution.reinit(n_dofs);
    last_solution.reinit(n_dofs);

    setup_done = true;
}

void CSFSolver::assemble_relaxation_step() {
    SparseMatrix<double> mass_matrix(sparsity_pattern);
    compute_system_matrices(current_solution, last_solution,
                            mass_matrix, relax_system_matrix);

    // The system matrix is 1/k M + K
    relax_system_matrix.add(1.0/time_step, mass_matrix);

    // The right hand side is 1/k M u_n
    mass_matrix.vmult(relax_rhs, last_solution);
    relax_rhs *= 1.0/time_step;

    // Apply boundary conditions
    if(time_dep_boundary_conditions || boundary_values.empty()) {
        boundary_function->set_time(current_time);
        VectorTools::interpolate_boundary_values(dof_handler, 0, *boundary_function, boundary_values);
    }
    MatrixTools::apply_boundary_values(boundary_values, relax_system_matrix, current_solution, relax_rhs);
}

double CSFSolver::weight_function(const Point<2> &pt) const {
    return 1.0;
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
                    // If there is a weight function, compute its value at the quad point
                    double weight_fn_value = weight_function(fe_values.quadrature_point(q_pt_idx));
                    // Compute the mass component
                    local_mass_matrix(i, j) += (fe_values.shape_value(i, q_pt_idx) * fe_values.shape_value(j, q_pt_idx) * weight_fn_value * grad_sum_inv * fe_values.JxW(q_pt_idx));
                    // Compute the stiffness component
                    local_stiffness_matrix(i, j) += (fe_values.shape_grad(i, q_pt_idx) * fe_values.shape_grad(j, q_pt_idx) * weight_fn_value * grad_sum_inv * fe_values.JxW(q_pt_idx));
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

void CSFSolver::run() {
    // Preconditions
    Assert(boundary_function != NULL, ExcInternalError());
    Assert(initial_condition != NULL, ExcInternalError());

    // Setup
    if(!setup_done) {
        setup_system();
    }

    // Initial condition
    ConstraintMatrix constraints;
    constraints.close();
    initial_condition->set_time(0.0);
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
    deallog.push("CSF Time Stepping");
    unsigned int total_steps = ceil(final_time / time_step);
    unsigned int step_idx = 1;
    while(current_time < final_time) {
        current_time += time_step;
        step_idx++;

        deallog << "=== (" << step_idx << ") Advancing time to " << current_time << " ===" << std::endl;

        assemble_relaxation_step();

        Vector<double> residual_vector(dof_handler.n_dofs());
        double residual = 0.0;
        unsigned int relaxation_steps = 0;
        do {
            // Verify that we haven't exceeded max steps
            if(relaxation_steps > max_relaxation_steps) {
                deallog << "Note relaxation_steps exceed max_relaxation_steps. Continuing to next time step." << std::endl;
                break;
            }

            solve_relaxation_step();

            assemble_relaxation_step();

            relax_system_matrix.vmult(residual_vector, current_solution);
            residual_vector -= relax_rhs;

            residual = residual_vector.l2_norm() / dof_handler.n_dofs();

            relaxation_steps++;

        } while(residual > relaxation_residual_tolerance);

        deallog << "Relaxation converged in " << relaxation_steps << " steps." << std::endl;

        last_solution = current_solution;

        write_output(step_idx);
    }
    deallog.pop();
}
