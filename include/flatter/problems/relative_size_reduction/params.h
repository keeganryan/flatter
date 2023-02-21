#pragma once

#include <vector>

#include <flatter/data/matrix.h>

namespace flatter {

class RelativeSizeReductionParams {
    // Size reduce basis B2 relative to basis B1
public:
    RelativeSizeReductionParams() {}
    RelativeSizeReductionParams(const Matrix& B1, const Matrix& B2, const Matrix& U);
    RelativeSizeReductionParams(const Matrix& B1, const Matrix& RV, const Matrix& tau, const Matrix& B2, const Matrix& U);

    bool is_B1_upper_triangular;
    int new_shift;

    Matrix B1;
    Matrix RV;
    Matrix R2;
    Matrix tau;
    Matrix B2;
    Matrix U;

private:
};

}