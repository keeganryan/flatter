#include "iterated.h"

#include <cassert>

#include "problems/qr_factorization.h"
#include "problems/size_reduction.h"
#include "problems/matrix_multiplication.h"

namespace flatter {
namespace FusedQRSizeRedImpl {

const std::string Iterated::impl_name() {return "Iterated";}

Iterated::Iterated(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

Iterated::~Iterated() {
    if (_is_configured) {
        unconfigure();
    }
}

void Iterated::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Iterated::configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(params, cc);
    assert(tau.nrows() == 0);
    assert(m == n);
    
    _is_configured = true;
}

void Iterated::solve() {
    log_start();

    Matrix U_sr(ElementType::MPZ, n, n);
    Matrix R_int(ElementType::MPZ, n, n, prec);
    U.set_identity();
    Matrix::copy(R, B);
    
    QRFactorization qr(R, cc);
    SizeReduction sr(R_int, U_sr, cc);
    qr.solve();

    while (true) {
        to_int_lattice(R_int);
        sr.solve();

        if (U_sr.is_identity()) {
            break;
        }
        
        // Update B and U
        MatrixMultiplication B_update(B, B, U_sr, cc);
        MatrixMultiplication U_update(U, U, U_sr, cc);
        B_update.solve();
        U_update.solve();

        Matrix::copy(R, B);
        qr.solve();
    }

    log_end();
}

void Iterated::to_int_lattice(const Matrix& R_int) {
    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    MatrixData<mpz_t> dR_int = R_int.data<mpz_t>();

    mpfr_t tmp;
    mpfr_init2(tmp, this->prec);

    double minval, maxval;
    double val;
    long exp;

    mpfr_abs(tmp, dR(0,0), rnd);
    val = mpfr_get_d_2exp(&exp, tmp, rnd);
    minval = exp + log2(val);
    maxval = minval;

    for (unsigned int i = 1; i < n; i++) {
        mpfr_abs(tmp, dR(i,i), rnd);
        val = mpfr_get_d_2exp(&exp, tmp, rnd);
        double logval = log2(val) + exp;
        minval = std::min(minval, logval);
        maxval = std::max(maxval, logval);
    }
    double spread = maxval - minval;

    unsigned int prec = spread + n + 10;
    int scale_exp = -1 * (maxval - prec);


    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mpfr_mul_2si(tmp, dR(i, j), scale_exp, rnd);
            mpfr_get_z(dR_int(i,j), tmp, rnd);
        }
        if (mpz_cmp_ui(dR_int(i,i), 0) == 0) {
            assert(false);
        }
    }

    mpfr_clear(tmp);
}

}
}