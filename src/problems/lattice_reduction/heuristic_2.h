#pragma once

#include "heuristic_3.h"

namespace flatter {
namespace LatticeReductionImpl {

class Heuristic2 : public Heuristic3 {
public:
    Heuristic2(const LatticeReductionParams& p, const ComputationContext& cc);

    const std::string impl_name();

    //void solve(void);
protected:
    std::vector<Matrix> B2s;
    std::vector<Matrix> U2s;

    unsigned int b2_cols;
    Matrix B2_orig;
    Matrix B_orig;
    Matrix B2_sim;

    virtual void init_solver();
    virtual void fini_solver();

    virtual bool is_reduced();
    virtual void setup_sublattice_reductions();
    virtual void cleanup_sublattice_reductions();

    virtual void init_compressed_B();
    virtual void update_representation();
    virtual void update_L_representation();
    virtual void update_R_representation();
    virtual void update_all_representation();

    virtual void fini_iter();
};

}
}