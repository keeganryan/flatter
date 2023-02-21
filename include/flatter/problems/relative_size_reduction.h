#pragma once

#include "relative_size_reduction/base.h"
#include "relative_size_reduction/params.h"

namespace flatter {

class RelativeSizeReduction : public RelativeSizeReductionImpl::Base {
public:
    RelativeSizeReduction();
    RelativeSizeReduction(const RelativeSizeReductionParams& params, const ComputationContext& cc);

    ~RelativeSizeReduction();

    void configure(const RelativeSizeReductionParams& params,
                    const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool _is_configured;
    RelativeSizeReductionImpl::Base* szred2;
};

}