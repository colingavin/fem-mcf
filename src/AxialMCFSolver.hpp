#ifndef AXIAL_MCF_SOLVER_INCLUDE
#define AXIAL_MCF_SOLVER_INCLUDE

#include "CSFSolver.hpp"

#define AXIAL_BOUNDARY_ID 1

class AxialMCFSolver : public virtual CSFSolver {
public:
    AxialMCFSolver();
    // Implement weighting for axial MCF
    virtual double weight_function(const Point<2> &pt) const;
    // Sets up the correct boundary_ids on the triangulation 
    // to give Neumann BCs on the axis -- call after make_grid
    void setup_boundary_conditions();
};

#endif
