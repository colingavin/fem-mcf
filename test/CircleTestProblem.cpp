#include "../src/CSFSolver.hpp"

#include <deal.II/base/logstream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

#define OUTPUT_TIME_EPSILON 1e-6

#define TEST_MIN_REFINEMENT 3
#define TEST_MAX_REFINEMENT 7
#define TEST_OUTPUT_TIME 0.125
#define TEST_SPATIAL_TIME_STEP (TEST_OUTPUT_TIME / 50)
#define TEST_START_TIME_STEP TEST_OUTPUT_TIME / 2
#define TEST_TEMPORAL_STEPS 5
#define TEST_TEMPORAL_SPACE_REFINEMENTS 6

using namespace dealii;

class CircleTestSolver : public CSFSolver {
public:
    CircleTestSolver(const double _output_time, 
                     const std::string _output_file_path,
                     const unsigned int _refinements) : CSFSolver() {
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

void CircleTestSolver::make_grid() {
    GridGenerator::hyper_cube(triangulation, -1.0, 1.0);

    triangulation.refine_global(refinements);

    deallog << "Made grid. Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

void CircleTestSolver::write_output(const unsigned int timestep_number) {
    if(std::abs(output_time - current_time) < OUTPUT_TIME_EPSILON) {
        deallog <<  "Time = " << current_time 
                << ". Writing solution to: " << output_file_path << std::endl;

        DataOut<2> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(current_solution, "u");
        data_out.build_patches();

        std::ofstream out(output_file_path);
        data_out.write_vtk(out);
    }
}

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
};

int main() {
    deallog.depth_console(1);

    // for(unsigned int refinement = TEST_MIN_REFINEMENT; 
    //     refinement <= TEST_MAX_REFINEMENT;
    //     refinement++) {
    //     CircleTestSolver solver(TEST_OUTPUT_TIME, 
    //         "circle-test/spatial-refinements-" + std::to_string(refinement) + ".vtk", 
    //         refinement);
    //     TestSolutionExact test_soln;
    //     solver.initial_condition = &test_soln;
    //     solver.boundary_function = &test_soln;
    //     solver.time_step = TEST_SPATIAL_TIME_STEP;
    //     solver.final_time = TEST_OUTPUT_TIME + OUTPUT_TIME_EPSILON;
    //     solver.run();
    // }

    for(unsigned int temporal_step = 0;
        temporal_step <= TEST_TEMPORAL_STEPS;
        temporal_step++) {
        CircleTestSolver solver(TEST_OUTPUT_TIME, 
            "circle-test/temporal-refinements-" + std::to_string(temporal_step) + ".vtk", 
            TEST_TEMPORAL_SPACE_REFINEMENTS);
        TestSolutionExact test_soln;
        solver.initial_condition = &test_soln;
        solver.boundary_function = &test_soln;
        solver.time_step = TEST_START_TIME_STEP * std::pow(2.0, -(double)temporal_step);
        solver.final_time = TEST_OUTPUT_TIME + OUTPUT_TIME_EPSILON;
        solver.run();
    }
}