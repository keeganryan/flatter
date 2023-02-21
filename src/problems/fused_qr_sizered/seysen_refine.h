#pragma once

#include <vector>

#include "problems/fused_qr_sizered/base.h"

namespace flatter {
namespace FusedQRSizeRedImpl {

class SeysenRefine : public Base {
public:
    SeysenRefine(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);
    ~SeysenRefine();

    const std::string impl_name();

    void configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);
    void solve(void);

    std::vector<unsigned int> split_list;

private:
    void size_reduce_column(unsigned int col);
    void size_reduce_columns();

    void unconfigure();

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

    Matrix sol;
    Matrix U_col_update;

    Matrix my_tau;
};

}
}