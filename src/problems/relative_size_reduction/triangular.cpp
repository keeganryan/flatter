#include "triangular.h"

#include <cassert>

namespace flatter {
namespace RelativeSizeReductionImpl {

const std::string Triangular::impl_name() {return "Triangular";}

Triangular::Triangular(const RelativeSizeReductionParams& params, const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

Triangular::~Triangular() {
    if (_is_configured) {
        unconfigure();
    }
}

void Triangular::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Triangular::configure(const RelativeSizeReductionParams& params, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    assert(params.B1.type() == ElementType::MPZ);
    assert(params.B2.type() == ElementType::MPZ);
    assert(params.U.type() == ElementType::MPZ);
    assert(params.is_B1_upper_triangular);
    
    Base::configure(params, cc);

    _is_configured = true;
}

void Triangular::solve() {
    log_start();

    MatrixData<mpz_t> dB1 = params.B1.data<mpz_t>();
    MatrixData<mpz_t> dB2 = params.B2.data<mpz_t>();
    MatrixData<mpz_t> dU = params.U.data<mpz_t>();

    mpz_t c_int, num, den;
    mpz_init(c_int);
    mpz_init(num);
    mpz_init(den);

    for (unsigned int j = 0; j < params.B2.ncols(); j++) {
        // Size reduce column j
        for (unsigned int i = 0; i < params.B2.nrows(); i++) {
            unsigned int row = params.B2.nrows() - i - 1;

            mpz_t& diag = dB1(row, row);
            mpz_t& entry = dB2(row, j);

            // We want c_int = round(entry / diag)
            //               = floor(entry / diag + 0.5)
            //               = floor((2*entry + diag) / 2*diag)
            mpz_mul_2exp(num, entry, 1);
            mpz_add(num, num, diag);
            mpz_mul_2exp(den, diag, 1);
            mpz_fdiv_q(c_int, num, den);
            if (mpz_cmp_ui(c_int, 0) != 0) {
                // Update B and U
                mpz_neg(dU(row, j), c_int);
                for (unsigned int k = 0; k <= row; k++) {
                    mpz_mul(num, dB1(k, row), c_int);
                    mpz_sub(dB2(k,j), dB2(k, j), num);
                }
            }

            if (params.new_shift < 0) {
                mpz_mul_2exp(dB2(row, j), dB2(row, j), -params.new_shift);
            } else {
                mpz_tdiv_q_2exp(dB2(row, j), dB2(row, j), params.new_shift);
            }
        }
    }

    if (params.R2.nrows() == params.B2.nrows() && params.R2.ncols() == params.B2.ncols()) {
        Matrix::copy(params.R2, params.B2);
    }

    mpz_clear(c_int);
    mpz_clear(num);
    mpz_clear(den);

    log_end();
}

}
}