#pragma once

#include "lattice_reduction/base.h"
#include <flatter/data/lattice.h>

namespace flatter {

class LatticeReduction : public LatticeReductionImpl::Base {
public:
    LatticeReduction();
    LatticeReduction(const LatticeReductionParams& p,
                    const ComputationContext& cc);
    LatticeReduction(const Matrix& M, const Matrix& U,
                    const ComputationContext& cc);
    LatticeReduction(Lattice L, Matrix& U,
                    const ComputationContext& cc);

    ~LatticeReduction();

    void configure(const LatticeReductionParams& p,
                    const ComputationContext& cc);
    void configure(const Matrix& M, const Matrix& U,
                    const ComputationContext& cc);
    void solve(void);

    static void set_policy(unsigned int policy);
    
private:
    void unconfigure();

    bool _is_configured;
    LatticeReductionImpl::Base* latred;

    static unsigned int policy;
};

}