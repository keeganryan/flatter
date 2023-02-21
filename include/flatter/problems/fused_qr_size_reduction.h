#pragma once

#include "fused_qr_sizered/base.h"

namespace flatter {

/**
 * This class implements a Fused QR Factorization and Size Reduction.
 * 
 * Input is a lattice basis B.
 * Output is in B, R, and U.
 *   B is a new, size-reduced basis
 *   R is a floating point QR factorization of the size-reduced basis.
 *   U is the unimodular matrix converting the old basis to new.
 * 
 */

class FusedQRSizeReduction : public FusedQRSizeRedImpl::Base {
public:
    FusedQRSizeReduction();
    FusedQRSizeReduction(
        const Matrix& B,
        const Matrix& R, const Matrix& U,
        const ComputationContext& cc);
    FusedQRSizeReduction(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);
    ~FusedQRSizeReduction();

    void configure(
        const Matrix& B,
        const Matrix& R, const Matrix& U,
        const ComputationContext& cc);
    void configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool _is_configured;
    FusedQRSizeRedImpl::Base* fqrszred;
};

}