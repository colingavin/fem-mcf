#include "SolverDiagnostics.hpp"

#include <deal.II/base/logstream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <fstream>

#define OUTPUT_TIME_EPSILON 1e-6

#define TEST_MIN_STEP 1
#define TEST_MAX_STEP 6
#define TEST_OUTPUT_TIME 0.1

using namespace dealii;

class DirichletTestSolver : public SolverDiagnosticsDifferencing {
public:
    DirichletTestSolver() : SolverDiagnosticsDifferencing() {}

    virtual void make_grid() {
        GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

        triangulation.refine_global(refinements);

        deallog << "Made grid. Number of active cells: " << triangulation.n_active_cells() << std::endl;
    }
};

class TestInitialCond : public Function<2> {
public:
    TestInitialCond() : Function<2, double>() {}
    virtual double value(const Point<2> &pt, const unsigned int component = 0) const {
        const double x = pt(0);
        const double y = pt(1);
        return 16 * x * (1 - x) * y * (1 - y);
    }
};

int main() {
    deallog.depth_console(1);

    DirichletTestSolver *previous_solver = NULL;
    DirichletTestSolver *solver = NULL;

    for(unsigned int refinement = TEST_MIN_STEP;
        refinement <= TEST_MAX_STEP;
        refinement++) {
        deallog << "Beginning test for refinement = " << refinement << std::endl;

        solver = new DirichletTestSolver();
        solver->output_time = TEST_OUTPUT_TIME;
        solver->output_file_path = "output/dirbc-test/refinements-" + std::to_string(refinement) + ".vtk";
        solver->refinements = refinement;

        TestInitialCond initial_condition;
        solver->initial_condition = &initial_condition;
        solver->boundary_function = &initial_condition;
        solver->time_step = TEST_OUTPUT_TIME * 0.02;
        solver->final_time = TEST_OUTPUT_TIME;
        solver->relaxation_residual_tolerance = 1e-8;
        solver->max_relaxation_steps = 20;

        solver->setup_and_interpolate(previous_solver);
        solver->run();

        deallog << "L2 Error = " << solver->output_l2_error << std::endl
                << "H1 Error = " << solver->output_h1_error << std::endl;

        previous_solver = solver;
    }
}
