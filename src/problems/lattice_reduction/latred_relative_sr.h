#pragma once

#include "problems/lattice_reduction/base.h"

namespace flatter {
namespace LatticeReductionImpl {

class LatRedRelSR : public Base {
public:
    LatRedRelSR(const LatticeReductionParams& p, const ComputationContext& cc);
    ~LatRedRelSR();

    const std::string impl_name();

    void configure(const LatticeReductionParams& p, const ComputationContext& cc);
    void solve(void);

private:
    bool _is_configured;

    void unconfigure();
};

}
}