#include "../src/CSFSolver.hpp"

#include <deal.II/base/logstream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

#include <nlopt.hpp>

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

class PaperclipTestSolver : public CSFSolver {
public:
    PaperclipTestSolver(const double _output_time, 
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

void PaperclipTestSolver::make_grid() {
    GridGenerator::hyper_cube(triangulation, -2.0, 2.0);

    triangulation.refine_global(refinements);

    deallog << "Made grid. Number of active cells: " << triangulation.n_active_cells() << std::endl;
}

void PaperclipTestSolver::write_output(const unsigned int timestep_number) {
    if(std::abs(output_time - current_time) < 1e-6) {
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

int main() {
    deallog.depth_console(1);

    PaperclipTestSolver solver(0.5, "paperclip-test/test.vtk", 6);
    solver.time_step = 0.5 / 30.0;
    solver.final_time = 0.5 + 1e-6;
    solver.time_dep_boundary_conditions = false;
    PaperclipSignedDistanceFn paperclip_fn;
    solver.boundary_function = &paperclip_fn;
    solver.initial_condition = &paperclip_fn;
    solver.run();
}
