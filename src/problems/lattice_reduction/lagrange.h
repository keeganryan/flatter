#pragma once

#include "problems/lattice_reduction/base.h"

#include "workspace_buffer.h"

namespace flatter {
namespace LatticeReductionImpl {

class Lagrange : public Base {
public:
    Lagrange(const LatticeReductionParams& p, const ComputationContext& cc);
    ~Lagrange();

    const std::string impl_name();

    void configure(const LatticeReductionParams& p, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    void xgcd(mpz_t& a, mpz_t& b, mpz_t& g, mpz_t& x, mpz_t& y);

    void norm2(mpfr_t& r, mpz_t& x1, mpz_t& x2, WorkspaceBuffer<mpfr_t>& ws);

    bool _is_configured;

    mpfr_rnd_t rnd;
};

}
}