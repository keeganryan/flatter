#pragma once

#include <vector>

#include <flatter/data/matrix.h>

namespace flatter {

class FusedQRSizeReductionParams {
public:
    FusedQRSizeReductionParams() {}
    FusedQRSizeReductionParams(const Matrix& B, const Matrix& R, const Matrix& U);
    FusedQRSizeReductionParams(const Matrix& B, const Matrix& R, const Matrix& U, const Matrix& tau);

    Matrix B() const {return B_;}
    Matrix R() const {return R_;}
    Matrix U() const {return U_;}
    Matrix tau() const {return tau_;}

    std::vector<unsigned int> prereduced_sublattice_inds;

private:
    Matrix B_;
    Matrix R_;
    Matrix U_;
    Matrix tau_;
};

}