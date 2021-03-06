//#include "../src/CSFSolver.hpp"
#include "SolverDiagnostics.hpp"

#include <deal.II/base/logstream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#define TEST_MIN_STEP 1
#define TEST_MAX_STEP 5
#define TEST_OUTPUT_TIME 0.1

using namespace dealii;

class CircleTestSolver : public SolverDiagnosticsExact {
public:
    CircleTestSolver() : SolverDiagnosticsExact() {}
    virtual void make_grid() {
        GridGenerator::hyper_cube(triangulation, -1.0, 1.0);

        triangulation.refine_global(refinements);

        deallog << "Made grid. Number of active cells: "
                << triangulation.n_active_cells() << std::endl;
    }
};

class TestSolutionExact : public Function<2> {
public:
    TestSolutionExact() : Function<2, double>() {}
    virtual double value(const Point<2> &pt, const unsigned int component = 0) const {
        const double x = pt(0);
        const double y = pt(1);
        const double t = get_time();
        const double r2 = std::pow(x, 2.0) + std::pow(y, 2.0);
        return std::exp(-t-r2/2.0) - std::exp(-0.5);
    }

    virtual Tensor<1, 2, double> gradient(const Point<2> &pt, const unsigned int component = 0) const {
        const double x = pt(0);
        const double y = pt(1);
        const double t = get_time();
        const double r2 = std::pow(x, 2.0) + std::pow(y, 2.0);
        Tensor<1, 2, double> result;
        result[0] = x;
        result[1] = y;
        result *= -std::exp(-t-r2/2.0);
        return result;
    }
};

int main() {
    deallog.depth_console(1);

    std::ofstream error_norms_out("circle-test/error_norms.csv");
    error_norms_out << "n, l2, h1" << std::endl;

    for(unsigned int refinement = TEST_MIN_STEP;
        refinement <= TEST_MAX_STEP;
        refinement++) {
        deallog << "Beginning test for refinement = " << refinement << std::endl;

        CircleTestSolver solver;
        solver.output_time = TEST_OUTPUT_TIME;
        solver.output_file_path = "output/circle-test/refinements-" + std::to_string(refinement) + ".vtk";
        solver.refinements = refinement;

        TestSolutionExact test_soln;
        solver.initial_condition = &test_soln;
        solver.boundary_function = &test_soln;
        solver.exact_soln = &test_soln;
        solver.time_step = TEST_OUTPUT_TIME * std::pow(2.0, -2.0*refinement);
        solver.final_time = TEST_OUTPUT_TIME;
        solver.relaxation_residual_tolerance = 1e-7;
        solver.max_relaxation_steps = 20;
        solver.grad_epsilon = 1e-6;
        solver.run();

        deallog << "L2 Error = " << solver.output_l2_error << std::endl
                << "H1 Error = " << solver.output_h1_error << std::endl;

        error_norms_out << refinement << ", "
                << solver.output_l2_error << ", "
                << solver.output_h1_error << std::endl;
    }
}
