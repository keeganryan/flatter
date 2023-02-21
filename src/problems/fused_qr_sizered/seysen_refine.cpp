#include "seysen_refine.h"

#include <cassert>

#include "problems/fused_qr_size_reduction.h"
#include "problems/qr_factorization.h"
#include "problems/size_reduction.h"
#include "problems/matrix_multiplication.h"

#include "math/mpfr_lapack.h"
#include "workspace_buffer.h"

namespace flatter {
namespace FusedQRSizeRedImpl {

const std::string SeysenRefine::impl_name() {return "SeysenRefine";}

SeysenRefine::SeysenRefine(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

SeysenRefine::~SeysenRefine() {
    if (_is_configured) {
        unconfigure();
    }
}

void SeysenRefine::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void SeysenRefine::configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(params, cc);
    
    _is_configured = true;
}


void SeysenRefine::solve() {
    log_start();

    U.set_identity();

    assert(this->params.prereduced_sublattice_inds.size() >= 1);
    std::vector<unsigned int> prereduced_sublattice_inds(this->params.prereduced_sublattice_inds);

    // Save tau and T so we can multiply by Q later
    if (tau.nrows() == 0) {
        my_tau = Matrix(ElementType::MPFR, n, 1, R.prec());
    } else {
        my_tau = tau;
    }

    split = prereduced_sublattice_inds.back();

    B_first = B.submatrix(0, m, 0, split);
    R_first = R.submatrix(0, m, 0, split);
    tau_first = my_tau.submatrix(0, split, 0, 1);
    U_first = U.submatrix(0, split, 0, split);

    R_right = R.submatrix(0, m, split, n);
    R_topright = R.submatrix(0, split, split, n);

    R_second = R.submatrix(split, m, split, n);
    tau_second = my_tau.submatrix(split, n, 0, 1);

    B_to_add = Matrix(ElementType::MPZ, m, n - split);
    sol = Matrix(ElementType::MPFR, split, cc.nthreads());
    U_col_update = Matrix(ElementType::MPZ, split, cc.nthreads());

    // Start by QR-factoring the first [0, split) vectors
    if (prereduced_sublattice_inds.size() > 1) {
        // Seysen Refine all but the last prereduced segment
        prereduced_sublattice_inds.pop_back();

        FusedQRSizeReductionParams params(B_first, R_first, U_first, tau_first);
        params.prereduced_sublattice_inds = prereduced_sublattice_inds;

        FusedQRSizeReduction prob(params, cc);
        prob.solve();
    } else {   
        QRFactorization qr_first(R_first, tau_first, cc);
        Matrix::copy(R_first, B.submatrix(0, m, 0, split));
        qr_first.solve();
    }

    size_reduce_columns();

    // Should be possible to get R just by doing QR
    QRFactorization qr_second(R_second, tau_second, cc);
    qr_second.solve();

    // Maybe clear subdiagonal

    if (tau.nrows() == 0) {
        clear_subdiagonal();
    }

    log_end();
}

void SeysenRefine::size_reduce_columns() {
    #pragma omp parallel for num_threads(cc.nthreads())
    for (unsigned int col = split; col < n; col++) {
        size_reduce_column(col);
    }
}

void SeysenRefine::size_reduce_column(unsigned int col) {
    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

    unsigned int lwork = 6;
    WorkspaceBuffer<mpfr_t> wsb(lwork + 2, prec);
    mpfr_t* work = wsb.walloc(lwork);
    mpfr_t* local = wsb.walloc(2);
    mpz_t c_int;
    mpz_t tmp_int;
    mpz_init(c_int);
    mpz_init(tmp_int);

    Matrix R_col = R.submatrix(0, m, col, col+1);
    Matrix B_col = B.submatrix(0, m, col, col+1);
    unsigned int tid = (unsigned int) omp_get_thread_num();
    Matrix my_sol = sol.submatrix(0, split, tid, tid+1);
    Matrix U_col = U_col_update.submatrix(0, split, tid, tid+1);

    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    MatrixData<mpfr_t> dR_col = R_col.data<mpfr_t>();
    MatrixData<mpfr_t> dsol = my_sol.data<mpfr_t>();
    MatrixData<mpfr_t> dV = R.data<mpfr_t>();
    MatrixData<mpfr_t> dTau = my_tau.data<mpfr_t>();
    MatrixData<mpz_t> dU_col = U_col.data<mpz_t>();
    MatrixData<mpz_t> dU = U.data<mpz_t>();
    MatrixData<mpz_t> dB = B.data<mpz_t>();

    for (unsigned int i = 0; i < split; i++) {
        mpz_set_ui(dU_col(i, 0), 0);
    }

    for (unsigned int repeats = 0; repeats < 1000; ) {
        // Update the floating point R-factor based on the exact B column
        Matrix::copy(R_col, B_col);
        for (unsigned int i = 0; i < split; i++) {
            // For each of the householder vectors
            larf(m - i, 1, &dV(i,i), dV.stride(), dTau(i,0), &dR_col(i, 0), dR_col.stride(), work, lwork);
        }

        // We have an approximately correct dR_col, so size reduce it

        // From algorithm 8.1 of Higham
        mpfr_t s;
        mpfr_t tmp;
        mpfr_init2(s, prec);
        mpfr_init2(tmp, prec);

        bool should_keep_reducing = false;
        for (unsigned int i = 0; i < split; i++) {
            unsigned int row = split - i - 1;
            if (i == 0) {
                // x_n = b_n // u_nn
                mpfr_div(dsol(row, 0), dR_col(row, 0), dR(row, row), rnd);
            } else {
                // s = b_i
                mpfr_set(s, dR_col(row, 0), rnd);

                for (unsigned int j = row + 1; j < split; j++) {
                    // s = s - u_ij x_j
                    mpfr_mul(tmp, dR(row, j), dsol(j, 0), rnd);
                    mpfr_sub(s, s, tmp, rnd);
                }

                // x_i = s / u_ii
                mpfr_div(dsol(row, 0), s, dR(row, row), rnd);
            }
        }
        for (unsigned int i = 0; i < split; i++) {
            unsigned int row = split - i - 1;
            // x_i is now fully determined
            // u_i,col is -round(x_i)
            mpfr_t& c = dsol(row, 0);
            mpfr_round(c, c);

            int exp = mpfr_get_exp(c);
            int bits_to_shift = std::max(0, exp - (int)prec);
            //bits_to_shift = 0;
            if (bits_to_shift > 0) {
                should_keep_reducing = true;
                //did_some_reduction = true;
            } else if (exp > (int)prec / 2) {
                should_keep_reducing = true;
            }

            // Divide by 2**bits_to_shift
            mpfr_div_2si(c, c, bits_to_shift, rnd);
            mpfr_get_z(c_int, c, rnd);
            if (bits_to_shift == 0 && mpz_cmp_ui(c_int, 0) == 0) {
                continue;
            }
            mpz_mul_2exp(c_int, c_int, bits_to_shift);

            // We now have a valid value of c_int
            // for this row. update dR_col, B_col,
            // and U_col
            for(unsigned int j = 0; j <= row; j++) {
                mpfr_mul_z(tmp, dR(j, row), c_int, rnd);
                mpfr_sub(dR_col(j, 0), dR_col(j, 0), tmp, rnd);
            }
            for (unsigned int j = 0; j < m; j++) {
                mpz_mul(tmp_int, dB(j, row), c_int);
                mpz_sub(dB(j, col), dB(j, col), tmp_int);
            }
            mpz_sub(dU_col(row, 0), dU_col(row, 0), c_int);
        }
        mpfr_clear(s);
        mpfr_clear(tmp);

        if (!should_keep_reducing) {
            break;
        }
    }

    // U_col now contains the accumulated updates we need to apply
    // to U.
    for (unsigned int i = 0; i < split; i++) {
        // Add U_col(i) copies of U(i) from col
        for (unsigned int j = 0; j <= i; j++) {
            // Take advantage of the fact that U is upper triangular
            mpz_mul(tmp_int, dU_col(i, 0), dU(j, i));
            mpz_add(dU(j, col), dU(j, col), tmp_int);
        }
    }

    wsb.wfree(local, 2);
    wsb.wfree(work, lwork);
    mpz_clear(c_int);
    mpz_clear(tmp_int);
}

void SeysenRefine::clear_subdiagonal() {
    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    // Clear below diagonal
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < i && j < n; j++) {
            mpfr_set_zero(dR(i,j), 0);
        }
    }
}

}
}