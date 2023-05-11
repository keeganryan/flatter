#pragma once

#include "heuristic_2.h"

namespace flatter {
namespace LatticeReductionImpl {

class Heuristic1 : public Heuristic2 {
public:
    Heuristic1(const LatticeReductionParams& p, const ComputationContext& cc);

    const std::string impl_name();
    
protected:
    virtual void init_compressed_B();
    virtual void setup_sublattice_reductions();

    virtual bool is_reduced();
    virtual void update_L_representation();
    virtual void update_R_representation();
    virtual void update_all_representation();

    int get_total_shift();
};

}
}