#pragma once

#include <fplll/fplll.h>

#include "problems/lattice_reduction/base.h"

namespace flatter {
namespace LatticeReductionImpl {

class FPLLL : public Base {
public:
    FPLLL(const LatticeReductionParams& p,
            const ComputationContext& cc);
    ~FPLLL();

    const std::string impl_name();

    void configure(const LatticeReductionParams& p,
                    const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    unsigned int get_block_size_for_rhf(double rhf);

    void init_A();
    bool _is_configured;

    fplll::ZZ_mat<mpz_t> A;
    mpfr_rnd_t rnd;
};

}
}