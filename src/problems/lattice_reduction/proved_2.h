#pragma once

#include "proved_3.h"

namespace flatter {
namespace LatticeReductionImpl {

class Proved2 : public Proved3 {
public:
    Proved2(const LatticeReductionParams& p, const ComputationContext& cc);
    
    const std::string impl_name();

protected:

    virtual bool is_reduced();
    virtual void setup_sublattice_reductions();

private:
};

}
}