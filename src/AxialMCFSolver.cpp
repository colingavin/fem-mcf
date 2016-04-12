#include "AxialMCFSolver.hpp"

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

AxialMCFSolver::AxialMCFSolver() : CSFSolver() {}

double AxialMCFSolver::weight_function(const Point<2> &pt) const {
    return pt(0);
}

void AxialMCFSolver::setup_boundary_conditions() {
    for(auto cell : triangulation.active_cell_iterators()) {
        for(unsigned int face_idx = 0; face_idx < GeometryInfo<2>::faces_per_cell; face_idx++) {
            TriaIterator<TriaAccessor<1, 2, 2>> f = cell->face(face_idx);
            if(f->center()[0] == 0.0) f->set_all_boundary_ids(AXIAL_BOUNDARY_ID);
        }
    }
}
