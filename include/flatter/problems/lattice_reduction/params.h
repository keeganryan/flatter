#pragma once

#include <flatter/data/matrix.h>
#include <flatter/data/lattice.h>
#include "goal.h"
#include "sublattice_split.h"

namespace flatter {

class LatticeReductionParams {
public:
    LatticeReductionParams();
    LatticeReductionParams(const Matrix& B, const Matrix& U, double rhf=1.03, bool is_upper_triangular=false);
    LatticeReductionParams(Lattice& L, Matrix& U, double rhf=1.03, bool is_upper_triangular=false);

    Matrix B() const {return L.basis();}
    Matrix U() const {return U_;}

    double rhf() const {return rhf_;}
    bool is_upper_triangular() const {return _is_upper_triangular;}

    bool proved;

    Lattice L;

    unsigned int phase;
    Matrix B2;
    Matrix U2;

    double* profile_offset;
    double log_cond;
    bool aggressive_precision;
    unsigned int offset;

    unsigned int lvalid;
    unsigned int rvalid;

    LatticeReductionGoal goal;
    SublatticeSplit* split;

private:
    Matrix U_;
    double rhf_;
    bool _is_upper_triangular;
};

}