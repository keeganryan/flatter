#include "elementary_ZZ.h"

#include <cassert>

#include "workspace_buffer.h"

namespace flatter {
namespace SizeReductionImpl {

const std::string ElementaryZZ::impl_name() {return "ElementaryZZ";}

ElementaryZZ::ElementaryZZ(const Matrix& R, const Matrix& U, const ComputationContext& cc) :
    Base(R, U, cc)
{
    _is_configured = false;
    configure(R, U, cc);
}

ElementaryZZ::~ElementaryZZ() {
    if (_is_configured) {
        unconfigure();
    }
}

void ElementaryZZ::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void ElementaryZZ::configure(const Matrix& R, const Matrix& U, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    assert(R.type() == ElementType::MPZ);
    assert(U.type() == ElementType::MPZ);
    
    Base::configure(R, U, cc);

    dR = R.data<mpz_t>();
    dU = U.data<mpz_t>();

    _is_configured = true;
}

void ElementaryZZ::solve() {
    log_start();

    // Set U to identity
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mpz_set_ui(dU(i,j), (i==j)?1:0);
        }
    }

    // Size reduce R, transform U by the same transformations
    mpz_t tmp[2], mu;
    mpz_init(tmp[0]);
    mpz_init(tmp[1]);
    mpz_init(mu);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int ind = 0; ind < i; ind++) {
            unsigned int j = i - ind - 1;
            mpz_div_round(mu, dR(j, i), dR(j, j), tmp);

            // Update R and U
            for (unsigned int k = 0; k < n; k++) {
                mpz_mul(tmp[0], mu, dR(k, j));
                mpz_sub(dR(k, i), dR(k, i), tmp[0]);
                mpz_mul(tmp[0], mu, dU(k, j));
                mpz_sub(dU(k, i), dU(k, i), tmp[0]);
            }
        }
    }

    mpz_clear(mu);
    mpz_clear(tmp[1]);
    mpz_clear(tmp[0]);

    log_end();
}

void ElementaryZZ::mpz_div_round(mpz_t& q, const mpz_t& a, const mpz_t& b, mpz_t* work) {
    // Return q = round(a / b)
    // We do this by
    // q = round(a / b) = floor(a / b + 0.5) = floor((2a+b)/2b)
    mpz_t& num = work[0];
    mpz_t& den = work[1];

    mpz_mul_2exp(num, a, 1);
    mpz_add(num, num, b);
    mpz_mul_2exp(den, b, 1);
    mpz_fdiv_q(q, num, den);
}

}
}