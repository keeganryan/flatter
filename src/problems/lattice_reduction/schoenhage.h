#pragma once

#include "problems/lattice_reduction/base.h"

namespace flatter {
namespace LatticeReductionImpl {

class Schoenhage : public Base {
public:
    Schoenhage(const LatticeReductionParams& p, const ComputationContext& cc);
    ~Schoenhage();

    const std::string impl_name();

    void configure(const LatticeReductionParams& p, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    void xgcd(mpz_t& a, mpz_t& b, mpz_t& g, mpz_t& x, mpz_t& y);

    void nonrecursive(mpz_t& a, mpz_t& b, mpz_t& c, unsigned int m, Matrix U);
    void recursive(mpz_t& a, mpz_t& b, mpz_t& c, unsigned int m, Matrix U);

    void simple_step(mpz_t& a, mpz_t& b, mpz_t& c, unsigned int m, mpz_t& t, bool& is_low_step);
    bool is_minimal(mpz_t& a, mpz_t& b, mpz_t& c, unsigned int m);

    bool _is_configured;
};

}
}