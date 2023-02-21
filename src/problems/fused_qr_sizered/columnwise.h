#pragma once

#include "problems/fused_qr_sizered/base.h"

namespace flatter {
namespace FusedQRSizeRedImpl {

class Columnwise : public Base {
public:
    Columnwise(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);
    ~Columnwise();

    const std::string impl_name();

    void configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    void to_int_lattice(const Matrix& R_int);

    bool _is_configured;
};

}
}