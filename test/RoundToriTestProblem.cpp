#include "../src/AxialMCFSolver.hpp"

#include <deal.II/base/logstream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

#define OUTPUT_TIME_EPSILON 1e-6

#define TEST_MIN_STEP 5
#define TEST_MAX_STEP 5
#define TEST_OUTPUT_TIME 0.1

using namespace dealii;

class ToriTestSolver : public AxialMCFSolver {
public:
    ToriTestSolver(const double _output_time,
                     const std::string _output_file_path,
                     const unsigned int _refinements) : AxialMCFSolver() {
        output_time = _output_time;
        output_file_path = _output_file_path;
        refinements = _refinements;
    }

    virtual void make_grid();
    virtual void write_output(const unsigned int timestep_number);

    // Time at which to output results
    double output_time;
    // Path to output
    std::string output_file_path;
    // Number of refinements of the grid
    unsigned int refinements;
};

void ToriTestSolver::make_grid() {
    Point<2> lower_left(0.0, -1.0);
    Point<2> upper_right(2.0, 1.0);
    GridGenerator::hyper_rectangle(triangulation, lower_left, upper_right);

    setup_boundary_conditions();

    triangulation.refine_global(refinements);

    deallog << "Made grid. Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

void ToriTestSolver::write_output(const unsigned int timestep_number) {
    if(timestep_number % 5 == 0) {
        deallog <<  "Time = " << current_time
                << ". Writing solution to: " << output_file_path << std::endl;

        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(current_solution, "u");
        data_out.build_patches();

        std::ofstream out(output_file_path + "-" + std::to_string(timestep_number) + ".vtk");
        data_out.write_vtk(out);
    }
}

class ToriInitialCondition : public Function<2> {
public:
    ToriInitialCondition() : Function<2, double>() {}
    virtual double value(const Point<2> &pt, const unsigned int component = 0) const {
        //1 - y^2 - x^2 (x - 1) (x + 1)
        double x = pt(0), y = pt(1);
        return 1 - std::pow(y, 2.0) - std::pow(x, 2.0) * (x - 1) * (x + 1);
        //return 1.0 - pt.distance(Point<2>(1.0, 0.0));
    }
};

int main() {
    deallog.depth_console(1);

    for(unsigned int refinement = TEST_MIN_STEP;
        refinement <= TEST_MAX_STEP;
        refinement++) {
        ToriTestSolver solver(TEST_OUTPUT_TIME,
            "output/tori-test/refinements-" + std::to_string(refinement),
            refinement);
        ToriInitialCondition ibc;
        solver.initial_condition = &ibc;
        solver.boundary_function = &ibc;
        solver.time_step = TEST_OUTPUT_TIME * std::pow(2.0, -2.0*refinement);
        solver.final_time = TEST_OUTPUT_TIME + OUTPUT_TIME_EPSILON;
        solver.relaxation_residual_tolerance = 1e-5;
        solver.max_relaxation_steps = 10;
        solver.time_dep_boundary_conditions = false;
        solver.run();
    }
}
