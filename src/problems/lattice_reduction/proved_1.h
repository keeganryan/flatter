#pragma once

#include "problems/lattice_reduction/base.h"

namespace flatter {
namespace LatticeReductionImpl {

class Proved1 : public Base {
public:
    Proved1(const LatticeReductionParams& p, const ComputationContext& cc);
    ~Proved1();

    const std::string impl_name();

    void configure(const LatticeReductionParams& p, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool _is_configured;
};

}
}