#ifndef SOLUTION_DIAGNOSTICS_INCLUDE

#define SOLUTION_DIAGNOSTICS_INCLUDE

#include "../src/CSFSolver.hpp"

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

class SolverDiagnostics : public virtual CSFSolver {
public:
    SolverDiagnostics();
    
    virtual void write_output(const unsigned int timestep_number);

    // Time at which to output results
    double output_time;
    // Path to output
    std::string output_file_path;
    // Number of refinements of the grid
    unsigned int refinements;

    // After run, holds l2 error at output time
    double output_l2_error;
    // After run, holds h1 error at output time
    double output_h1_error;

protected:
    virtual double current_l2_error() = 0;
    virtual double current_h1_error() = 0;
};

// Computes final errors by comparison to an exact solution
class SolverDiagnosticsExact : public SolverDiagnostics {
public:
  SolverDiagnosticsExact();

  Function<2> *exact_soln;

protected:
    virtual double current_l2_error();
    virtual double current_h1_error();
};

// Computes final errors by differencing with a previous coarser step
class SolverDiagnosticsDifferencing : public SolverDiagnostics {
public:
    SolverDiagnosticsDifferencing();
    void setup_and_interpolate(SolverDiagnosticsDifferencing *previous_solver);

private:
    Vector<double> coarse_solution_interpolated;

protected:
    virtual double current_l2_error();
    virtual double current_h1_error();
};

#endif
