#include "SolverDiagnostics.hpp"

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>

#define OUTPUT_TIME_EPSILON 1e-10

// Abstract SolverDiagnostics constructor
SolverDiagnostics::SolverDiagnostics() : CSFSolver() {}

void SolverDiagnostics::write_output(const unsigned int timestep_number) {
    if(std::abs(output_time - current_time) < OUTPUT_TIME_EPSILON) {
        // Write current solution
        deallog <<  "Time = " << current_time 
                << ". Writing solution to: " << output_file_path << std::endl;

        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(current_solution, "u");
        data_out.build_patches();

        std::ofstream out(output_file_path);
        data_out.write_vtk(out);
        
        // Compute errors
        output_l2_error = current_l2_error();
        output_h1_error = current_h1_error();
    }
}

// SolverDiagnosticsExact implementation

SolverDiagnosticsExact::SolverDiagnosticsExact() : SolverDiagnostics() {}

double SolverDiagnosticsExact::current_l2_error() {
    exact_soln->set_time(current_time);
    Vector<double> l2_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(
        dof_handler, 
        current_solution,
        *exact_soln,
        l2_per_cell,
        QGauss<2>(3),
        VectorTools::L2_norm);
    return l2_per_cell.l2_norm();
}

double SolverDiagnosticsExact::current_h1_error() {
    exact_soln->set_time(current_time);
    Vector<double> h1_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(
        dof_handler, 
        current_solution,
        *exact_soln,
        h1_per_cell,
        QGauss<2>(3),
        VectorTools::H1_norm);

    return h1_per_cell.l2_norm();
}

// SolverDiagnosticsDifferencing implementation

SolverDiagnosticsDifferencing::SolverDiagnosticsDifferencing() : SolverDiagnostics () {}

void SolverDiagnosticsDifferencing::setup_and_interpolate(SolverDiagnosticsDifferencing *previous_solver) {
    setup_system();
    
    coarse_solution_interpolated.reinit(dof_handler.n_dofs());

    if(previous_solver != NULL) {
        VectorTools::interpolate_to_different_mesh(
            previous_solver->dof_handler,
            previous_solver->current_solution,
            dof_handler,
            coarse_solution_interpolated);
    }
}

double SolverDiagnosticsDifferencing::current_l2_error() {
    SparseMatrix<double> mass_matrix(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<2>(3), mass_matrix);

    Vector<double> difference = current_solution;
    difference -= coarse_solution_interpolated;

    return mass_matrix.matrix_norm_square(difference);
}

double SolverDiagnosticsDifferencing::current_h1_error() {
    SparseMatrix<double> mass_matrix(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler, QGauss<2>(3), mass_matrix);

    SparseMatrix<double> stiffness_matrix(sparsity_pattern);
    MatrixCreator::create_laplace_matrix(dof_handler, QGauss<2>(3), stiffness_matrix);
    stiffness_matrix.add(1.0, mass_matrix);

    Vector<double> difference = current_solution;
    difference -= coarse_solution_interpolated;

    return stiffness_matrix.matrix_norm_square(difference);
}


