#pragma once

#include "size_reduction/base.h"

namespace flatter {

class SizeReduction : public SizeReductionImpl::Base {
public:
    SizeReduction();
    SizeReduction(const Matrix& M, const Matrix& U,
                    const ComputationContext& cc);

    ~SizeReduction();

    void configure(const Matrix& M, const Matrix& U,
                    const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool _is_configured;
    SizeReductionImpl::Base* szred;
};

}