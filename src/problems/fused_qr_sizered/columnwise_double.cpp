#include "columnwise_double.h"

#include <cassert>

#include "problems/qr_factorization.h"
#include "problems/size_reduction.h"
#include "problems/matrix_multiplication.h"

#include "workspace_buffer.h"

#include "columnwise.h"

typedef int lapack_int;
extern "C" {
    int dlarfg_(int *, double *, double *, int *, double *);
    int dlarf_(const char *, int *, int *, double *, int *, double *, double *, int *, double *);
}

namespace flatter {
namespace FusedQRSizeRedImpl {

const std::string ColumnwiseDouble::impl_name() {return "ColumnwiseDouble";}

ColumnwiseDouble::ColumnwiseDouble(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

ColumnwiseDouble::~ColumnwiseDouble() {
    if (_is_configured) {
        unconfigure();
    }
}

void ColumnwiseDouble::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void ColumnwiseDouble::configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(params, cc);

    assert(R.type() == ElementType::DOUBLE);
    assert(tau.nrows() == 0);

    _is_configured = true;
}

void ColumnwiseDouble::solve() {
    log_start();


    unsigned int lwork = n + 6;
    unsigned int alloc_size = n + lwork + 2;
    double* local;
    WorkspaceBuffer<double>* wsb = new WorkspaceBuffer<double>(alloc_size, R.prec());
    double* tau_ptr = wsb->walloc(n);
    local = wsb->walloc(2);
    double* work = wsb->walloc(lwork);
    MatrixData<double> dR = R.data<double>();
    double& entry = local[0];
    double& c = local[1];
    mpz_t c_int, prod;
    mpz_init(c_int);
    mpz_init(prod);
    MatrixData<mpz_t> dB = B.data<mpz_t>();
    U.set_identity();
    MatrixData<mpz_t> dU = U.data<mpz_t>();

    long int exp_shift = 0;
    for (unsigned int i = 0; i < n; i++) {
        bool must_repeat = false;
        double mu_max = INFINITY;
        do {
            // Copy column from B to R

            // Copy column from B to R, scaling to avoid infinity
            for (unsigned int i2 = 0; i2 < m; i2++) {
                long int exp;
                double v = mpz_get_d_2exp(&exp, dB(i2, i));
                if (exp + exp_shift > 1000) {
                    // This will round to infinity
                    long int delta = 1000 - (exp + exp_shift);
                    // Scale everything down from before this point
                    double s = pow(2, delta);
                    for (unsigned int i3 = 0; i3 < i; i3++) {
                        for (unsigned int i4 = 0; i4 <= i3; i4++) {
                            dR(i4, i3) *= s;
                        }
                    }
                    for (unsigned int i3 = 0; i3 < i2; i3++) {
                        dR(i3, i) *= s;
                    }
                    exp_shift += delta;
                }
                dR(i2, i) = v * pow(2, exp + exp_shift);
                assert(std::isfinite(dR(i2, i)));
            }

            // Apply orthogonal transformation of previous columns
            int l_arg1, l_arg2, l_arg3, l_arg4;
            for (unsigned int j = 0; j < i; j++) {
                l_arg1 = 1;
                l_arg2 = m - j;
                l_arg3 = dR.stride();
                l_arg4 = l_arg3;
                double tmp = dR(j,j);
                dR(j,j) = 1;
                dlarf_("Right", &l_arg1, &l_arg2, &dR(j,j), &l_arg3, &tau_ptr[j], &dR(j, i), &l_arg4, work);
                dR(j,j) = tmp;
            }

            // Check if size reduced
            must_repeat = false;
            double round_mu_max = 0;
            for (unsigned int j = 0; j < i; j++) {
                bool not_size_reduced = false;
                entry = dR(i-j-1, i);
                if (fabs(entry) >= fabs(dR(i - j - 1, i - j - 1))) {
                    // We have the entry is greater than 1 * diagonal, but
                    // really we should check if entry is greater than 0.51 * diagonal
                    not_size_reduced = true;
                } else {
                    entry = entry * 4;
                    if (fabs(entry) < fabs(dR(i - j - 1, i - j - 1))) {
                        not_size_reduced = false;
                    } else {
                        // 2 times entry
                        double entry_d = dR(i-j-1, i);
                        double diag_d = dR(i-j-1, i-j-1);

                        if (fabs(entry_d / diag_d) > 0.51) {
                            not_size_reduced = true;
                        }
                    }
                }
                if (not_size_reduced) {
                    // Not size reduced
                    must_repeat = true;
\
                    c = dR(i - j - 1, i) / dR(i - j - 1, i - j - 1);
                    c = round(c);
                    double mu = fabs(c);
                    round_mu_max = std::max(round_mu_max, mu);
                    mpz_set_d(c_int, c);

                    // Subtract column i-j-1 from column i
                    for (unsigned int k = 0; k < m; k++) {
                        mpz_mul(prod, c_int, dB(k, i-j-1));
                        mpz_sub(dB(k, i), dB(k, i), prod);
                    }
                    for (unsigned int k = 0; k < n; k++) {
                        mpz_mul(prod, c_int, dU(k, i-j-1));
                        mpz_sub(dU(k, i), dU(k, i), prod);

                        if (k < i - j) {
                            entry = c * dR(k, i-j-1);
                            dR(k, i) = dR(k, i) - entry;
                        }
                    }
                }
            }
            if (std::isfinite(round_mu_max) && round_mu_max > mu_max - (prec / 2)) {
                // Not making any more progress
                must_repeat = false;
            }
            mu_max = round_mu_max;
        } while (must_repeat);

        // Create householder vector with the remainder
        lapack_int arg_1;
        lapack_int arg_2;
        arg_1 = m - i;
        arg_2 = dR.stride();
        if (i < m - 1) {
            dlarfg_(&arg_1, &dR(i,i), &dR(i+1,i), &arg_2, &tau_ptr[i]);
        }
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int i2 = 0; i2 <= i; i2++) {
            // Scale back
            dR(i, i2) *= pow(2, -exp_shift);
        }
    }
    
    tau_ptr[n - 1] = 0;
    
    // Clear below diagonal
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < i && j < n; j++) {
            dR(i,j) = 0;
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

void ColumnwiseDouble::to_int_lattice(const Matrix& R_int) {
    MatrixData<double> dR = R.data<double>();
    MatrixData<mpz_t> dR_int = R_int.data<mpz_t>();

    double tmp;
    tmp = this->prec;

    double minval, maxval;
    double val;

    tmp = fabs(dR(0,0));
    val = tmp;
    minval = log2(val);
    maxval = minval;

    for (unsigned int i = 1; i < n; i++) {
        double logval = log2(fabs(dR(i,i)));
        minval = std::min(minval, logval);
        maxval = std::max(maxval, logval);
    }
    double spread = maxval - minval;

    unsigned int prec = spread + n + 10;
    int scale_exp = -1 * (maxval - prec);


    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            tmp = dR(i,j) * pow(2, scale_exp);
            mpz_set_d(dR_int(i,j), tmp);
        }
        if (mpz_cmp_ui(dR_int(i,i), 0) == 0) {
            assert(false);
        }
    }
}

}
}