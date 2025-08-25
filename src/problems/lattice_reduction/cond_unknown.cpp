#include "cond_unknown.h"

#include <algorithm>
#include <cassert>
#include <limits>

#include "math/mpfr_lapack.h"

#include "problems/lattice_reduction.h"
#include "problems/matrix_multiplication.h"
#include "problems/qr_factorization.h"
#include "problems/size_reduction.h"
#include "problems/relative_size_reduction.h"

#include "sublattice_split_2.h"
#include "workspace_buffer.h"

namespace flatter {
namespace LatticeReductionImpl {

const std::string CondUnknown::impl_name() {return "CondUnknown";}

CondUnknown::CondUnknown(const LatticeReductionParams& p, const ComputationContext& cc) :
    Base(p, cc)
{
    _is_configured = false;
    configure(p, cc);
}

CondUnknown::~CondUnknown() {
    if (_is_configured) {
        unconfigure();
    }
}

void CondUnknown::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void CondUnknown::configure(const LatticeReductionParams& p, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(p, cc);
    assert(M.type() == ElementType::MPZ);
    assert(U.type() == ElementType::MPZ);
    assert(params.B2.ncols() == 0);

    _is_configured = true;
}

void CondUnknown::apply_U(const Matrix& U1, const Matrix& U2) {
    unsigned int k = U1.nrows();
    Matrix B_left = B.submatrix(0, B.nrows(), 0, k);
    Matrix B_right;
    if (k < B.ncols()) {
        B_right = B.submatrix(0, B.nrows(), k, B.ncols());

        MatrixMultiplication mm1 (B_right, B_left, U2, true, cc);
        mm1.solve();
    }

    MatrixMultiplication mm2 (B_left, B_left, U1, false, cc);
    mm2.solve();
}

void CondUnknown::apply_perm(const Matrix& U) {
    MatrixData<mpz_t> dU = U.data<mpz_t>();
    Matrix B2(ElementType::MPZ, B.nrows(), B.ncols());

    for (unsigned int j = 0; j < U.ncols(); j++) {
        // Find where the vector should come from
        unsigned int src;
        for (src = 0; src < B.ncols(); src++) {
            if (mpz_cmp_ui(dU(src, j), 1) == 0) {
                break;
            }
        }
        assert(src < B.ncols());

        Matrix::copy(
            B2.submatrix(0, B.nrows(), j, j+1),
            B.submatrix(0, B.nrows(), src, src+1)
        );
    }
    Matrix::copy(B, B2);
}

void CondUnknown::extract_similar(const Matrix& B, unsigned int prec, Matrix Bsim, Matrix U, unsigned int& num_valid, int& shift_amount, double& spread) {
    // Given a collection of input vectors and a desired precision, find Bsim, U such that Bsim is prec-similar to BU,
    // and there are numvalid linearly independent vectors at the beginning of B.
    Matrix R(ElementType::MPFR, B.nrows(), B.ncols(), prec);
    Matrix R_unsorted(ElementType::MPFR, B.nrows(), B.ncols(), prec);
    U.set_identity();
    Matrix::copy(R_unsorted, B);

    unsigned int max_rank = std::min(B.nrows(), B.ncols());
    unsigned int num_dependent = 0;
    unsigned int lwork = 6;
    
    WorkspaceBuffer<mpfr_t> ws(3 + max_rank + lwork, prec);
    mpfr_t* local = ws.walloc(3);
    mpfr_t* tau_ptr = ws.walloc(max_rank);
    mpfr_t* work = ws.walloc(lwork);

    mpfr_t& tmp = local[0];
    mpfr_t& vec_len = local[1];
    mpfr_t& orth_len = local[2];
    double log_vec_len;
    double log_orth_len;
    double curmax = -INFINITY;
    double curmin = INFINITY;

    num_valid = 0;

    // Find the next column that has a large enough orthogonal component
    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    MatrixData<mpfr_t> dRu = R_unsorted.data<mpfr_t>();
    MatrixData<mpz_t> dU = U.data<mpz_t>();
    for (unsigned int j = 0; j < R.ncols(); j++) {
        mpfr_set_zero(vec_len, 0);
        mpfr_set_zero(orth_len, 0);

        for (unsigned int i = 0; i < R.nrows(); i++) {
            mpfr_sqr(tmp, dRu(i, j), rnd);
            mpfr_add(vec_len, vec_len, tmp, rnd);
            if (i >= num_valid) {
                mpfr_add(orth_len, orth_len, tmp, rnd);
            }
        }

        long exp;
        double d = mpfr_get_d_2exp(&exp, vec_len, rnd);
        log_vec_len = (exp + log2(fabs(d))) / 2;
        d = mpfr_get_d_2exp(&exp, orth_len, rnd);
        log_orth_len = (exp + log2(fabs(d))) / 2;

        double new_spread = std::max(curmax, log_vec_len) - std::min(curmin, log_orth_len);
        unsigned int required_prec = 2 * new_spread + 40;
        if (!std::isinf(new_spread) && !std::isnan(new_spread) && required_prec <= prec) {
            // Copy from R_unsorted to R
            Matrix::copy(R.submatrix(0, R.nrows(), num_valid, num_valid+1), R_unsorted.submatrix(0, R.nrows(), j, j+1));
            mpz_set_ui(dU(j,j), 0);
            mpz_set_ui(dU(j,num_valid), 1);

            // Householder this column
            // Apply householder transformation to remaining unsorted vectors
            if (num_valid == R.nrows() - 1) {
                mpfr_set_zero(tau_ptr[num_valid], 0);
            } else {
                larfg(R.nrows() - num_valid, dR(num_valid, num_valid), &dR(num_valid + 1, num_valid), dR.stride(), tau_ptr[num_valid], work, lwork);

                if (j < R.ncols() - 1) {
                    larf(R.nrows() - num_valid, dR.ncols() - 1 - j, &dR(num_valid, num_valid), dR.stride(), tau_ptr[num_valid],
                        &dRu(num_valid, j + 1), dRu.stride(),
                        work, lwork
                    );
                }
            }
            num_valid++;
            curmax = std::max(curmax, log_vec_len);
            curmin = std::min(curmin, log_orth_len);
        } else {
            Matrix::copy(
                R.submatrix(0, R.nrows(), R.ncols() - num_dependent - 1, R.ncols() - num_dependent),
                R_unsorted.submatrix(0, R.nrows(), j, j+1)
            );
            mpz_set_ui(dU(j,j), 0);
            mpz_set_ui(dU(j, R.ncols() - num_dependent - 1), 1);
            num_dependent++;
        }
    }

    ws.wfree(work, lwork);
    ws.wfree(tau_ptr, max_rank);
    ws.wfree(local, 3);

    shift_amount = (int)prec - (int)curmax;
    spread = curmax - curmin;
    // Scale R appropriately
    for (unsigned int i = 0; i < dR.nrows(); i++) {
        for (unsigned int j = 0; j < dR.ncols(); j++) {
            if (i <= j) {
                mpfr_mul_2si(dR(i,j), dR(i,j), shift_amount, rnd);
            } else {
                mpfr_set_ui(dR(i,j), 0, rnd);
            }
        }
    }
    Matrix::copy(Bsim, R);
}

void CondUnknown::sort_by_size(Matrix B) {
    Matrix U_sort(ElementType::MPZ, B.ncols(), B.ncols());
    /* Begin by sorting the columns of B from smallest to largest. */
    std::vector<double> log_vec_lens(B.ncols());

    MatrixData<mpz_t> dB = B.data<mpz_t>();
    mpz_t s, tmp;
    mpz_init(s);
    mpz_init(tmp);
    for (unsigned int j = 0; j < B.ncols(); j++) {
        mpz_set_ui(s, 0);
        for (unsigned int i = 0; i < B.nrows(); i++) {
            mpz_mul(tmp, dB(i,j), dB(i,j));
            mpz_add(s, s, tmp);
        }
        long exp;
        double d = mpz_get_d_2exp(&exp, s);
        log_vec_lens[j] = (exp + log2(d)) / 2;
    }
    mpz_clear(s);
    mpz_clear(tmp);

    // Fill U_step as a permutation matrix based on order of smallest elements. This is inefficient,
    // but bases should be small enough that this isn't a bottleneck
    MatrixData<mpz_t> dU = U_sort.data<mpz_t>();
    unsigned int end_idx = B.ncols() - 1;
    unsigned int start_idx = 0;
    for (unsigned int i = 0; i < B.ncols(); i++) {
        auto it = std::min_element(std::begin(log_vec_lens), std::end(log_vec_lens));
        unsigned int j = std::distance(std::begin(log_vec_lens), it);
        if (log_vec_lens[j] == -INFINITY) {
            // Vector is 0, put it at the end
            mpz_set_ui(dU(j, end_idx), 1);
            end_idx--;
        } else {
            // Vector is small, put it at the front
            mpz_set_ui(dU(j, start_idx), 1);
            start_idx++;
        }
        log_vec_lens[j] = INFINITY;
    }

    // Update B to sort from shortest vector to longest
    apply_perm(U_sort);
}

bool CondUnknown::refine_basis() {
    Matrix B_sim(ElementType::MPZ, B.nrows(), B.ncols());
    Matrix U_step(ElementType::MPZ, B.ncols(), B.ncols());
    Matrix U_1(ElementType::MPZ, B.ncols(), B.ncols());
    Matrix U_2(ElementType::MPZ, B.ncols(), B.ncols());
    Matrix U_3(ElementType::MPZ, B.ncols(), B.ncols());
    unsigned int num_valid;
    double spread;
    int shift_amount;

    extract_similar(B, working_prec, B_sim, U_1, num_valid, shift_amount, spread);

    // Do lattice reduction on vectors
    Matrix B_sim_indep = B_sim.submatrix(0, num_valid, 0, num_valid);
    Matrix B_sim_dep;
    if (num_valid < B_sim.ncols()) {
        B_sim_dep = B_sim.submatrix(0, num_valid, num_valid, B_sim.ncols());
    }

    // Do size reduction on vectors
    U_2.set_identity();
    Matrix U_2i = U_2.submatrix(0, num_valid, 0, num_valid);
    SizeReduction sr(B_sim_indep, U_2i, cc);
    sr.solve();

    Matrix U_2d;
    if (0 < num_valid && num_valid < B_sim.ncols()) {
        U_2d = U_2.submatrix(0, num_valid, num_valid, B_sim.ncols());
        RelativeSizeReductionParams rsr_params (B_sim_indep, B_sim_dep, U_2d);
        RelativeSizeReduction rsr(rsr_params, cc);
        rsr.solve();

        MatrixMultiplication mmsr (U_2d, U_2i, U_2d, cc);
        mmsr.solve();
    }

    U_3.set_identity();
    Matrix U_3i = U_3.submatrix(0, num_valid, 0, num_valid);
    Matrix U_3d;
    if (num_valid < B_sim.ncols()) {
        U_3d = U_3.submatrix(0, num_valid, num_valid, U_3.ncols());
    }

    LatticeReductionParams lr_params(B_sim_indep, U_3i, rhf, true);
    lr_params.proved = this->params.proved;
    lr_params.phase = 2;
    lr_params.split = new SubSplitPhase2(num_valid);
    lr_params.aggressive_precision = this->params.aggressive_precision;
    lr_params.B2 = B_sim_dep;
    lr_params.U2 = U_3d;
    lr_params.log_cond = spread;
    lr_params.profile_offset = new double[num_valid];
    for (unsigned int i = 0; i < num_valid; i++) {
        lr_params.profile_offset[i] = -shift_amount;
    }

    LatticeReduction lr(lr_params, cc);
    lr.solve();
    delete lr_params.split;
    delete[] lr_params.profile_offset;

    for (unsigned int i = 0; i < num_valid; i++) {
        this->params.L.profile[i] = lr_params.L.profile[i] - shift_amount;
    }

    apply_perm(U_1);
    if (num_valid > 0) {
        apply_U(U_2i, U_2d);
        apply_U(U_3i, U_3d);
    }

    // How many zero vectors are there?
    MatrixData<mpz_t> dB = B.data<mpz_t>();
    unsigned int zero_vecs = 0;
    for (unsigned int j = num_valid; j < B.ncols(); j++) {
        bool vec_is_zero = true;
        for (unsigned int i = 0; i < B.nrows(); i++) {
            if (mpz_sgn(dB(i,j)) != 0) {
                vec_is_zero = false;
                break;
            }
        }
        if (vec_is_zero) {
            zero_vecs++;
        }
    }

    if (num_valid + zero_vecs == B.ncols()) {
        return true;
    }
    if (num_valid == max_rank) {
        working_prec = 2 * this->params.L.profile.get_spread() + 40;
        return false;
    }
    max_rank = std::min(B.nrows(), B.ncols() - zero_vecs);
    working_prec *= 2;
    return false;
}

void CondUnknown::solve() {
    log_start();

    mon->profile_reset(std::min(m, n));

    working_prec = 53;
    max_rank = std::min(M.nrows(), M.ncols());
    B = Matrix(ElementType::MPZ, M.nrows(), M.ncols());
    R = Matrix(ElementType::MPFR, M.nrows(), M.ncols(), working_prec);
    Matrix::copy(B, M);

    U.set_identity();

    for (unsigned int i = 0; i < 10000; i++) {
        if (refine_basis()) {
            break;
        }
        sort_by_size(B);
    }

    Matrix::copy(M, B);
    this->params.L.update_rank();

    log_end();
}

}
}