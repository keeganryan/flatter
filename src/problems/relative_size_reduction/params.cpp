#include "problems/relative_size_reduction/params.h"

namespace flatter {

RelativeSizeReductionParams::RelativeSizeReductionParams(const Matrix& B1, const Matrix& B2, const Matrix& U) :
    RelativeSizeReductionParams(B1, Matrix(), Matrix(), B2, U)
{}

RelativeSizeReductionParams::RelativeSizeReductionParams(const Matrix& B1, const Matrix& RV, const Matrix& tau, const Matrix& B2, const Matrix& U) :
    B1(B1),
    RV(RV),
    tau(tau),
    B2(B2),
    U(U)
{
    is_B1_upper_triangular = false;
    new_shift = 0;
    if (RV.nrows() != 0) {
        R2 = Matrix(RV.type(), B2.nrows(), B2.ncols(), RV.prec());
    }
}

}