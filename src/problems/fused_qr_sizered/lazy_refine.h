#pragma once

#include <vector>

#include "problems/fused_qr_sizered/base.h"

namespace flatter {
namespace FusedQRSizeRedImpl {

/**
 * Fused QR/size reduction procedure based on the L2 algorithm's
 * lazy size reduction. If we have a lattice where we know two
 * sublattices are size reduced, each vector in the later
 * sublattice can be size reduced by the vectors in the first
 * sublattice in parallel. We use floating point, even though
 * the sublattice vectors can be not size reduced by a lot.
 */

class LazyRefine : public Base {
public:
    LazyRefine(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);
    ~LazyRefine();

    const std::string impl_name();

    void configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);
    void solve(void);

    std::vector<unsigned int> split_list;

private:
    void unconfigure();

    void size_reduce_col_partial(const Matrix& B_col, const Matrix &R_col, const Matrix& U_tmp);
    void size_reduce_column(unsigned int col);
    void size_reduce_columns();
    
    void clear_subdiagonal();

    bool _is_configured;

    unsigned int split;

    Matrix B_to_add;

    Matrix B_first;
    Matrix R_first;
    Matrix tau_first;
    Matrix U_first;

    Matrix R_second;
    Matrix tau_second;

    Matrix R_right;
    Matrix R_topright;

    Matrix TmpZ;
    Matrix TmpZ2;

    Matrix my_tau;
};

}
}