#pragma once

#include "problems/relative_size_reduction/base.h"

namespace flatter {
namespace RelativeSizeReductionImpl {

class Generic : public Base {
public:
    Generic(const RelativeSizeReductionParams& params, const ComputationContext& cc);
    ~Generic();

    const std::string impl_name();

    void configure(const RelativeSizeReductionParams& params, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool _is_configured;
};

}
}