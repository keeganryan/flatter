#include "problems/fused_qr_sizered/params.h"

namespace flatter {

FusedQRSizeReductionParams::FusedQRSizeReductionParams(const Matrix& B, const Matrix& R, const Matrix& U) :
    FusedQRSizeReductionParams(B, R, U, Matrix())
{}

FusedQRSizeReductionParams::FusedQRSizeReductionParams(const Matrix& B, const Matrix& R, const Matrix& U, const Matrix& tau) :
    B_(B),
    R_(R),
    U_(U),
    tau_(tau)
{}

}