#pragma once

#include "problems/relative_size_reduction/base.h"

namespace flatter {
namespace RelativeSizeReductionImpl {

class Triangular : public Base {
public:
    Triangular(const RelativeSizeReductionParams& params, const ComputationContext& cc);
    ~Triangular();

    const std::string impl_name();

    void configure(const RelativeSizeReductionParams& params, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool _is_configured;
};

}
}