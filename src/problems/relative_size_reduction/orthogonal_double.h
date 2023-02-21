#pragma once

#include "problems/relative_size_reduction/base.h"

namespace flatter {
namespace RelativeSizeReductionImpl {

class OrthogonalDouble : public Base {
public:
    OrthogonalDouble(const RelativeSizeReductionParams& params, const ComputationContext& cc);
    ~OrthogonalDouble();

    const std::string impl_name();

    void configure(const RelativeSizeReductionParams& params, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();
    
    void size_reduce(Matrix R, Matrix B, Matrix r_col, Matrix u_col, Matrix b_col, Matrix tau);

    bool _is_configured;
};

}
}