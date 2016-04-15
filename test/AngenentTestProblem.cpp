#include "../src/AxialMCFSolver.hpp"
#include "SolverDiagnostics.hpp"

#include <deal.II/base/logstream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

#define X_START 0.0
#define X_END 3.5
#define Y_START -1.0
#define Y_END 1.0
#define N_X_SAMPLES 351
#define N_Y_SAMPLES 201
#define SAMPLES_PATH "test/ang_distance_field_neumann.csv"

#define TEST_MIN_STEP 2
#define TEST_MAX_STEP 6
#define TEST_OUTPUT_TIME 0.2

class AngenentDistanceFn : public Function<2> {
public:
  AngenentDistanceFn() : Function<2, double>(),
    samples(N_X_SAMPLES * N_Y_SAMPLES) {
      std::ifstream sfile(SAMPLES_PATH);
      std::string line;
      unsigned int idx = 0;
      while(getline(sfile, line)) {
        samples[idx] = std::stod(line);
        idx++;
      }
  }

  virtual double value(const Point<2> &pt, const unsigned int component = 0) const {
    const double x = pt(0);
    const double y = pt(1);

    if(x < X_START || x > X_END || y < Y_START || y > Y_END) {
      deallog << "AngenentDistanceFn asked to extrapolate." << std::endl;
      abort();
    }
    double dx = (X_END - X_START)/(N_X_SAMPLES - 1);
    double dy = (Y_END - Y_START)/(N_Y_SAMPLES - 1);
    unsigned int xidx = std::floor((x - X_START)/dx);
    if(x == X_END) xidx--;
    unsigned int yidx = std::floor((y - Y_START)/dy);
    if(y == Y_END) yidx--;
    double ll = samples[xidx * N_Y_SAMPLES + yidx];
    double lr = samples[xidx * N_Y_SAMPLES + (yidx + 1)];
    double ul = samples[(xidx + 1) * N_Y_SAMPLES + yidx];
    double ur = samples[(xidx + 1) * N_Y_SAMPLES + (yidx + 1)];
    double w = x - (X_START + xidx * dx);
    double z = y - (Y_START + yidx * dy);
    return ll + (lr - ll)*w/dx + (ul - ll)*z/dy + (ll + ur - lr - ul)*w*z/(dx*dy);
  }

private:
  std::vector<double> samples;
};

class AngenentTestSolver : public SolverDiagnosticsDifferencing, public AxialMCFSolver {
public:
  AngenentTestSolver() : SolverDiagnosticsDifferencing() {}

  virtual void make_grid() {
    GridGenerator::hyper_rectangle(triangulation, Point<2>(X_START, Y_START), Point<2>(X_END, Y_END));

    setup_boundary_conditions();

    triangulation.refine_global(refinements);

    deallog << "Made grid. Number of active cells: " << triangulation.n_active_cells() << std::endl;
  }
};

int main(int argc, char const *argv[]) {
  deallog.depth_console(1);

  AngenentTestSolver *previous_solver = NULL;
  AngenentTestSolver *solver = NULL;

  for(unsigned int refinement = TEST_MIN_STEP;
      refinement <= TEST_MAX_STEP;
      refinement++) {
      deallog << "Beginning test for refinement = " << refinement << std::endl;

      solver = new AngenentTestSolver();
      solver->output_time = TEST_OUTPUT_TIME;
      solver->output_file_path = "output/angenent-test/refinements-" + std::to_string(refinement) + ".vtk";
      solver->refinements = refinement;

      AngenentDistanceFn test_soln;
      solver->initial_condition = &test_soln;
      solver->boundary_function = &test_soln;
      solver->time_dep_boundary_conditions = false;
      solver->time_step = TEST_OUTPUT_TIME * std::pow(2.0, -2.0*refinement);
      solver->final_time = TEST_OUTPUT_TIME;
      solver->relaxation_residual_tolerance = 1e-7;
      solver->max_relaxation_steps = 20;
      solver->grad_epsilon = 1e-6;

      solver->setup_and_interpolate(previous_solver);

      solver->run();

      deallog << "L2 Error = " << solver->output_l2_error << std::endl
              << "H1 Error = " << solver->output_h1_error << std::endl;

      previous_solver = solver;
  }
  return 0;
}
