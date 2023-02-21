#pragma once

#include "problems/size_reduction/base.h"

namespace flatter {
namespace SizeReductionImpl {

class ElementaryZZ : public Base {
public:
    ElementaryZZ(const Matrix& R, const Matrix& U, const ComputationContext& cc);
    ~ElementaryZZ();

    const std::string impl_name();

    void configure(const Matrix& R, const Matrix& U, const ComputationContext& cc);
    void solve(void);

    static void mpz_div_round(mpz_t& q, const mpz_t& a, const mpz_t& b, mpz_t* work);

private:
    void unconfigure();

    bool _is_configured;
    MatrixData<mpz_t> dR;
    MatrixData<mpz_t> dU;
};

}
}