#include "elementary_ll.h"

#include <cassert>

#include "workspace_buffer.h"

namespace flatter {
namespace SizeReductionImpl {

const std::string ElementaryLL::impl_name() {return "ElementaryLL";}

ElementaryLL::ElementaryLL(const Matrix& R, const Matrix& U, const ComputationContext& cc) :
    Base(R, U, cc)
{
    _is_configured = false;
    configure(R, U, cc);
}

ElementaryLL::~ElementaryLL() {
    if (_is_configured) {
        unconfigure();
    }
}

void ElementaryLL::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void ElementaryLL::configure(const Matrix& R, const Matrix& U, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    assert(R.type() == ElementType::INT64);
    assert(U.type() == ElementType::INT64);
    assert(!cc.is_threaded());
    
    Base::configure(R, U, cc);

    dR = R.data<int64_t>();
    dU = U.data<int64_t>();

    _is_configured = true;
}

void ElementaryLL::solve() {
    log_start();

    // Set U to identity
    U.set_identity();

    // Size reduce R, transform U by the same transformations
    int64_t mu;

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int ind = 0; ind < i; ind++) {
            unsigned int j = i - ind - 1;
            mu = round(double(dR(j, i)) / dR(j, j));

            // Update R and U
            for (unsigned int k = 0; k < n; k++) {
                dR(k, i) -= mu * dR(k, j);
                dU(k, i) -= mu * dU(k, j);
            }
        }
    }

    log_end();
}

void ElementaryLL::mpz_div_round(mpz_t& q, const mpz_t& a, const mpz_t& b, mpz_t* work) {
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