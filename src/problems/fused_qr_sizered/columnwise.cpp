#include "columnwise.h"

#include <cassert>

#include "problems/qr_factorization.h"
#include "problems/size_reduction.h"
#include "problems/matrix_multiplication.h"

#include "math/mpfr_lapack.h"
#include "workspace_buffer.h"

namespace flatter {
namespace FusedQRSizeRedImpl {

const std::string Columnwise::impl_name() {return "Columnwise";}

Columnwise::Columnwise(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

Columnwise::~Columnwise() {
    if (_is_configured) {
        unconfigure();
    }
}

void Columnwise::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Columnwise::configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(params, cc);

    assert(R.type() == ElementType::MPFR);
    assert(tau.nrows() == 0);

    _is_configured = true;
}

void Columnwise::solve() {
    log_start();


    unsigned int lwork = 6;
    unsigned int alloc_size = n + lwork + 2;
    mpfr_t* local;
    WorkspaceBuffer<mpfr_t>* wsb = new WorkspaceBuffer<mpfr_t>(alloc_size, R.prec());
    mpfr_t* tau_ptr = wsb->walloc(n);
    local = wsb->walloc(2);
    mpfr_t* work = wsb->walloc(lwork);
    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    mpfr_t& entry = local[0];
    mpfr_t& c = local[1];
    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
    mpz_t c_int, prod;
    mpz_init(c_int);
    mpz_init(prod);
    MatrixData<mpz_t> dB = B.data<mpz_t>();
    U.set_identity();
    MatrixData<mpz_t> dU = U.data<mpz_t>();

    // Copy column from B to R
    Matrix::copy(R, B);

    for (unsigned int i = 0; i < n; i++) {
        bool must_repeat = false;
        double mu_max = INFINITY;
        do {
            // Check if size reduced
            must_repeat = false;
            double round_mu_max = 0;
            for (unsigned int j = 0; j < i; j++) {
                bool not_size_reduced = false;
                mpfr_set(entry, dR(i-j-1, i), rnd);
                if (mpfr_cmp_abs(entry, dR(i - j - 1, i - j - 1)) >= 0) {
                    // We have the entry is greater than 1 * diagonal, but
                    // really we should check if entry is greater than 0.51 * diagonal
                    not_size_reduced = true;
                } else {
                    mpfr_mul_2si(entry, entry, 2, rnd);
                    if (mpfr_cmp_abs(entry, dR(i - j - 1, i - j - 1)) < 0) {
                        not_size_reduced = false;
                    } else {
                        // 2 times entry
                        double entry_d = mpfr_get_d(dR(i-j-1, i), rnd);
                        double diag_d = mpfr_get_d(dR(i-j-1, i-j-1), rnd);

                        if (fabs(entry_d / diag_d) > 0.51) {
                            not_size_reduced = true;
                        }
                    }
                }
                if (not_size_reduced) {
                    // Not size reduced
                    must_repeat = true;

                    mpfr_div(c, dR(i - j - 1, i), dR(i - j - 1, i - j - 1), rnd);
                    mpfr_round(c, c);
                    double mu = fabs(mpfr_get_d(c, rnd));
                    round_mu_max = std::max(round_mu_max, mu);
                    mpfr_get_z(c_int, c, rnd);

                    // Subtract column i-j-1 from column i
                    for (unsigned int k = 0; k < m; k++) {
                        mpz_mul(prod, c_int, dB(k, i-j-1));
                        mpz_sub(dB(k, i), dB(k, i), prod);
                    }
                    for (unsigned int k = 0; k < n; k++) {
                        mpz_mul(prod, c_int, dU(k, i-j-1));
                        mpz_sub(dU(k, i), dU(k, i), prod);

                        if (k < i - j) {
                            mpfr_mul(entry, c, dR(k, i-j-1), rnd);
                            mpfr_sub(dR(k, i), dR(k, i), entry, rnd);
                        }
                    }
                }
            }
            if (std::isfinite(round_mu_max) && round_mu_max >= mu_max - (prec / 2)) {
                // Not making any more progress
                must_repeat = false;
            }
            mu_max = round_mu_max;
            if (must_repeat) {
                // Re orthogonalize
                // Copy column from B to R
                Matrix::copy(R.submatrix(0,m,i,i+1), B.submatrix(0,m,i,i+1));

                // Apply orthogonal transformation of previous columns
                for (unsigned int j = 0; j < i; j++) {
                    larf(m-j, 1, &dR(j,j), dR.stride(), tau_ptr[j], &dR(j, i), dR.stride(), work, lwork);
                }
            }
        } while (must_repeat);

        // Create householder vector with the remainder
        if (i < m - 1) {
            larfg(m - i, dR(i,i), &dR(i+1,i), dR.stride(), tau_ptr[i], work, lwork);

            // Apply orthogonal transformation of previous columns
            #pragma omp taskloop if (cc.nthreads() > 1)
            for (unsigned int i2 = i + 1; i2 < n; i2++) {
                WorkspaceBuffer<mpfr_t>* wsb_t = new WorkspaceBuffer<mpfr_t>(lwork, R.prec());
                mpfr_t* work_t = wsb_t->walloc(lwork);
                larf(m-i, 1, &dR(i,i), dR.stride(), tau_ptr[i], &dR(i, i2), dR.stride(), work_t, lwork);
                wsb_t->wfree(work_t, lwork);
                delete wsb_t;
            }
        }
    }
    
    mpfr_set_zero(tau_ptr[n - 1], 0);
    
    // Clear below diagonal
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < i && j < n; j++) {
            mpfr_set_zero(dR(i,j), 0);
        }
    }

    wsb->wfree(work, lwork);
    wsb->wfree(local, 2);
    wsb->wfree(tau_ptr, n);
    mpz_clear(c_int);
    mpz_clear(prod);
    delete wsb;

    log_end();
}

void Columnwise::to_int_lattice(const Matrix& R_int) {
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