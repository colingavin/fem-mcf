#include "SolverDiagnostics.hpp"

#include <deal.II/base/logstream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

#include <nlopt.hpp>

#define TEST_MIN_STEP 1
#define TEST_MAX_STEP 5
#define TEST_OUTPUT_TIME 0.5

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

double paperclip_distance(const std::vector<double> &x, std::vector<double> &grad, void* data) {
    Point<2> pt = *(Point<2>*)(data);
    //std::cout << "Eval distance from x = " << x[0] << ", " << x[1] << " to = " << pt << std::endl;
    if(!grad.empty()) {
        //std::cout << "dist + grad" << std::endl;
        grad[0] = 2.0 * (x[0] - pt[0]);
        grad[1] = 2.0 * (x[1] - pt[1]);
    }
    return std::pow(x[0] - pt[0], 2.0) + std::pow(x[1] - pt[1], 2.0);
}

double paperclip_constraint(const std::vector<double> &x, std::vector<double> &grad, void* data) {
    double t = *((double*)data);
    //std::cout << "Eval constraint t = " << t << "at x = " << x[0] << ", " << x[1] << std::endl;
    if(!grad.empty()) {
        //std::cout << "constraint + grad" << std::endl;
        grad[0] = std::exp(1 - t) * std::sin(x[0]);
        grad[1] = std::sinh(x[1]);
    }
    return std::cosh(x[1]) - std::exp(1 - t) * std::cos(x[0]);
}

// Implements the signed distance function for the paperclip Cosh[y] - Exp[1 - t] Cos[x]
// where t in [0, 1] (vanishes at t = 1)
class PaperclipSignedDistanceFn : public Function<2> {
public:
    PaperclipSignedDistanceFn() : Function<2, double>() {}

    virtual double value(const Point<2> &pt, const unsigned int component = 0) const {
        double t = get_time();

        // Do nonlinear optimization to find closest point and distance
        nlopt::opt optimizer(nlopt::LD_SLSQP, 2);
        optimizer.set_min_objective(&paperclip_distance, (void*)&pt);
        optimizer.add_equality_constraint(&paperclip_constraint, (void*)&t);

        optimizer.set_ftol_abs(1e-5);
        optimizer.set_maxeval(1e3);

        double x_start = pt[0] >= 0 ? 2.2 : -2.2;
        std::vector<double> nearest_out = {x_start, 0.0};
        double dist_sq_out = 0.0;
        optimizer.optimize(nearest_out, dist_sq_out);

        // Determine the signed distance
        double displace_x = nearest_out[0] - pt[0];
        double displace_y = nearest_out[1] - pt[1];
        double normal_x = std::sinh(nearest_out[1]);
        double normal_y = -std::exp(1 - t) * std::sin(nearest_out[0]);

        double det = displace_x * normal_y - displace_y * normal_x;

        return std::sqrt(dist_sq_out) * sgn(det);
    }
};

class PaperclipTestSolver : public SolverDiagnosticsDifferencing {
public:
    PaperclipTestSolver() : SolverDiagnosticsDifferencing() {}

    virtual void make_grid() {
        GridGenerator::hyper_cube(triangulation, -2.0, 2.0);

        triangulation.refine_global(refinements);

        deallog << "Made grid. Number of active cells: " << triangulation.n_active_cells() << std::endl;
    }
};

int main() {
    deallog.depth_console(1);

    PaperclipTestSolver *previous_solver = NULL;
    PaperclipTestSolver *solver = NULL;

    for(unsigned int refinement = TEST_MIN_STEP; 
        refinement <= TEST_MAX_STEP;
        refinement++) {
        deallog << "Beginning test for refinement = " << refinement << std::endl;

        solver = new PaperclipTestSolver();
        solver->output_time = TEST_OUTPUT_TIME;
        solver->output_file_path = "paperclip-test/refinements-" + std::to_string(refinement) + ".vtk";
        solver->refinements = refinement;
        
        PaperclipSignedDistanceFn test_soln;
        solver->initial_condition = &test_soln;
        solver->boundary_function = &test_soln;
        solver->time_dep_boundary_conditions = false;
        solver->time_step = TEST_OUTPUT_TIME * std::pow(2.0, -2.0*refinement);
        solver->final_time = TEST_OUTPUT_TIME;
        solver->use_scheduled_relaxation = false;
        solver->relaxation_residual_tolerance = 1e-7;
        solver->max_relaxation_steps = 20;

        solver->setup_and_interpolate(previous_solver);

        solver->run();

        deallog << "L2 Error = " << solver->output_l2_error << std::endl
                << "H1 Error = " << solver->output_h1_error << std::endl;

        //if(previous_solver != NULL) delete previous_solver;
        
        previous_solver = solver;
    }
}
