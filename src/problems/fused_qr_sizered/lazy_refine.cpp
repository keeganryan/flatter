#include "lazy_refine.h"

#include <cassert>

#include "problems/qr_factorization.h"
#include "problems/size_reduction.h"
#include "problems/matrix_multiplication.h"

#include "problems/fused_qr_size_reduction.h"

#include "math/mpfr_lapack.h"
#include "workspace_buffer.h"

namespace flatter {
namespace FusedQRSizeRedImpl {

const std::string LazyRefine::impl_name() {return "LazyRefine";}

LazyRefine::LazyRefine(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

LazyRefine::~LazyRefine() {
    if (_is_configured) {
        unconfigure();
    }
}

void LazyRefine::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void LazyRefine::configure(
        const FusedQRSizeReductionParams& params,
        const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(params, cc);
    
    _is_configured = true;
}

void LazyRefine::solve() {
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

    TmpZ = Matrix(ElementType::MPZ, split, 2*cc.nthreads());
    TmpZ2 = Matrix(ElementType::MPZ, m, cc.nthreads());

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

void LazyRefine::size_reduce_columns() {
    #pragma omp taskgroup
    for (unsigned int col = split; col < n; col++) {
        #pragma omp task firstprivate(col) if (cc.nthreads() > 1)
        size_reduce_column(col);
    }
}

void LazyRefine::size_reduce_col_partial(const Matrix& B_col, const Matrix &R_col, const Matrix& U_tmp) {

    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

    unsigned int lwork = 6;
    WorkspaceBuffer<mpfr_t> wsb(lwork + 2, prec);
    mpfr_t* work = wsb.walloc(lwork);
    mpfr_t* local = wsb.walloc(2);
    mpfr_t& c = local[0];
    mpfr_t& tmp = local[1];
    mpz_t c_int;
    mpz_t tmp_int;
    mpz_init(c_int);
    mpz_init(tmp_int);

    //Matrix _mu = Matrix::alloc(ElementType::MPZ, split, 1, 0);
    //Matrix _muf = Matrix::alloc(ElementType::MPFR, split, 1, prec);

    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    MatrixData<mpfr_t> dR_col = R_col.data<mpfr_t>();
    MatrixData<mpfr_t> dV = R.data<mpfr_t>();
    MatrixData<mpfr_t> dTau = my_tau.data<mpfr_t>();
    MatrixData<mpz_t> dU_tmp = U_tmp.data<mpz_t>();
    MatrixData<mpz_t> dB = B.data<mpz_t>();
    MatrixData<mpz_t> dB_col = B_col.data<mpz_t>();
    //MatrixData<mpz_t> dmu = _mu.data<mpz_t>();
    //MatrixData<mpfr_t> dmuf = _muf.data<mpfr_t>();

    /*
    if (n == 146 && prec == 334 && col == 120) {
        printf("split %u\n", split);
        Matrix::save(B, "b.lat");
    }*/

    for (unsigned int i = 0; i < split; i++) {
        mpz_set_ui(dU_tmp(i, 0), 0);
    }

    for (unsigned int repeats = 0; repeats < 1000; ) {
        // Update the floating point R-factor based on the exact B column
        Matrix::copy(R_col, B_col);
        {
            // Apply householder vectors to b

            unsigned int prev = 0;
            for (auto it = params.prereduced_sublattice_inds.begin(); it != params.prereduced_sublattice_inds.end(); it++) {
                for (unsigned int i = prev; i < *it; i++) {
                    larf(m - i, 1, &dV(i,i), dV.stride(), dTau(i,0), &dR_col(i, 0), dR_col.stride(), work, lwork);
                }
                prev = *it;
            }
        }

        // We have an approximately correct dR_col, so size reduce it
        bool use_all_bits = true;
        // since the significand is considered in [0.5, 1),
        // any mu_size above -1 will not be size reduced.
        int max_mu_size = -1;
        unsigned int num_splits = params.prereduced_sublattice_inds.size();
        for (unsigned int split_ind = 0; split_ind < num_splits; split_ind ++) {
            unsigned int low_bound, high_bound;
            if (split_ind == num_splits - 1) {
                low_bound = 0;
            } else {
                low_bound = params.prereduced_sublattice_inds.at(num_splits - 1 - split_ind - 1);
            }
            high_bound = params.prereduced_sublattice_inds.at(num_splits - 1 - split_ind);

            for (unsigned int i = 0; i < high_bound - low_bound; i++) {
                unsigned int row = high_bound - i - 1;

                mpfr_t& entry = dR_col(row, 0);
                mpfr_t& diag = dR(row, row);

                mpfr_div(c, entry, diag, rnd);
                //mpfr_set(dmuf(row, 0), c, rnd);
                //mpfr_round(c, c);
                if (mpfr_cmp_d(c, -0.51) > 0 && mpfr_cmp_d(c, 0.51) < 0) {
                //if (mpfr_sgn(c) == 0) {
                    // c is zero
                    //mpz_set_ui(dmu(row, 0), 0);
                    continue;
                } else {
                    // Only get the current precision
                    int exp = mpfr_get_exp(c);
                    max_mu_size = std::max(exp, max_mu_size);
                    mpfr_get_z(c_int, c, rnd);
                }

                // We now have a valid value of c_int
                // for this row. update dR_col, B_col,
                // and U_tmp
                //mpz_set(dmu(row, 0), c_int);
                for(unsigned int j = 0; j <= row; j++) {
                    mpfr_mul_z(tmp, dR(j, row), c_int, rnd);
                    mpfr_sub(dR_col(j, 0), dR_col(j, 0), tmp, rnd);
                }
                for (unsigned int j = 0; j < m; j++) {
                    mpz_mul(tmp_int, dB(j, row), c_int);
                    mpz_sub(dB_col(j, 0), dB_col(j, 0), tmp_int);
                }
                mpz_sub(dU_tmp(row, 0), dU_tmp(row, 0), c_int);
            }
        }
        //Matrix::print(_muf);
        if (use_all_bits && max_mu_size <= 1) {
            break;
        }
    }

    // Check to see if condition is satisfied

    mpfr_t tmp1;
    mpfr_init2(tmp1, prec);

    mpfr_clear(tmp1);

    wsb.wfree(local, 2);
    wsb.wfree(work, lwork);
    mpz_clear(c_int);
    mpz_clear(tmp_int);
}

void LazyRefine::size_reduce_column(unsigned int col) {
    Matrix R_col = R.submatrix(0, m, col, col+1);
    Matrix B_col = B.submatrix(0, m, col, col+1);
    Matrix U_col = U.submatrix(0, n, col, col+1);
    unsigned int tid = 0;
    if (cc.nthreads() > 1) {
        tid = (unsigned int) omp_get_thread_num();
    }
    Matrix tmp_u = TmpZ.submatrix(0, split, 2*tid, 2*tid+1);
    Matrix u_0 = TmpZ.submatrix(0, split, 2*tid+1, 2*tid+2);
    Matrix b_0 = TmpZ2.submatrix(0, m, tid, tid+1);

    MatrixData<mpz_t> dB = B.data<mpz_t>();

    MatrixData<mpz_t> db = B_col.data<mpz_t>();
    MatrixData<mpz_t> db_0 = b_0.data<mpz_t>();

    MatrixData<mpz_t> dtmp_u = tmp_u.data<mpz_t>();
    MatrixData<mpz_t> du_0 = u_0.data<mpz_t>();

    mpz_t tmp_int;
    mpz_init(tmp_int);

    // We consider vector b in multiple chunks of k bits.
    // b = 2^p * b0 + b1
    // b0 - Bu0 is small
    // 2^p * b0 - B * 2^p * u0 is small
    // 2^p * b0 + b1 - B * 2^p * u0 is small

    unsigned int total_bits = 0;
    unsigned int basis_bits = 0;
    for (unsigned int i = 0; i < m; i++) {
        total_bits = std::max((unsigned int)mpz_sizeinbase(db(i,0), 2), total_bits);
        for (unsigned int j = 0; j < split; j++) {
            basis_bits = std::max((unsigned int)mpz_sizeinbase(dB(i,j), 2), basis_bits);
        }
    }
    unsigned int k = std::max(prec * 6, 300u);
    // Get next lower multiple of k
    int p = ((total_bits - 1 - basis_bits) / k) * k;
    p = std::max(0, p);
    p = 0;
    k = 0;

    for (unsigned int i = 0; i < m; i++) {
        mpz_set_ui(db_0(i, 0), 0);
    }
    for (unsigned int i = 0; i < split; i++) {
        mpz_set_ui(dtmp_u(i, 0), 0);
    }
    while (p >= 0) {
        // Shift b_0 up by k bits
        for (unsigned int i = 0; i < m; i++) {
            mpz_mul_2exp(db_0(i,0), db_0(i,0), k);
        }
        // Shift tmp_u up by k bits
        for (unsigned int i = 0; i < split; i++) {
            mpz_mul_2exp(dtmp_u(i, 0), dtmp_u(i, 0), k);
        }

        // Add bits above 2**p in b_col to b_short, remove from b_col
        for (unsigned int i = 0; i < m; i++) {
            mpz_tdiv_q_2exp(tmp_int, db(i, 0), p);
            mpz_add(db_0(i, 0), db_0(i, 0), tmp_int);
            mpz_tdiv_r_2exp(db(i, 0), db(i, 0), p);
        }

        // Reduce vector b by B, update rotation R[col], store update in u
        // Find u such that b + Bu is size reduced.
        size_reduce_col_partial(b_0, R_col, u_0);

        for (unsigned int i = 0; i < split; i++) {
            mpz_add(dtmp_u(i, 0), dtmp_u(i, 0), du_0(i, 0));
        }
        break;
    }

    // Should be fully reduced now
    Matrix::copy(B_col, b_0);

    // U_col now contains the accumulated updates we need to apply
    // to U.
    MatrixData<mpz_t> dU_col = U_col.data<mpz_t>();
    MatrixData<mpz_t> dU = U.data<mpz_t>();
    for (unsigned int i = 0; i < split; i++) {
        // Add U_col(i) copies of U(i) from col
        for (unsigned int j = 0; j <= i; j++) {
            // Take advantage of the fact that U is upper triangular
            mpz_mul(tmp_int, dtmp_u(i, 0), dU(j, i));
            mpz_add(dU_col(j, 0), dU_col(j, 0), tmp_int);
        }
    }
    mpz_clear(tmp_int);
}

void LazyRefine::clear_subdiagonal() {
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