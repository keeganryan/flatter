#include "tri_matmul.h"

#include <cassert>
#include <mpfr.h>

#include "flatter/monitor.h"

namespace flatter {

TriMatrixMultiplication::TriMatrixMultiplication(
                            MatrixData<mpfr_t>& A, MatrixData<mpfr_t>& B,
                            char side, char uplo, char transa, char diag,
                            const mpfr_t* alpha,
                            WorkspaceBuffer<mpfr_t>* wsb,
                            const ComputationContext& cc) :
    A(A), B(B),
    side(side), uplo(uplo), transa(transa), diag(diag), 
    alpha(alpha), wsb(wsb), cc(&cc)
{
    work = wsb->walloc(2);
}

TriMatrixMultiplication::~TriMatrixMultiplication() {
    wsb->wfree(work, 2);
}

void TriMatrixMultiplication::solve() {
    assert(side == 'L' || side == 'R');
    assert(uplo == 'U' || uplo == 'L');
    assert(transa == 'N' || transa == 'T');
    assert(diag == 'N' || diag == 'U');

    unsigned int m = B.nrows();
    unsigned int n = B.ncols();
    unsigned int ldb = B.stride();
    unsigned int lda = A.stride();
    mpfr_t* b_dat = B.get_data();
    mpfr_t* a_dat  = A.get_data();
    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

    mpfr_t& tmp = work[0];
    mpfr_t& prod = work[1];

    assert(
        (side == 'R' && uplo == 'L' && transa == 'N' && diag == 'U') ||
        (side == 'R' && uplo == 'L' && transa == 'T' && diag == 'U') ||
        (side == 'R' && uplo == 'U' && transa == 'N' && diag == 'N')
    );

    if (side == 'R' && uplo == 'L' && transa == 'N' && diag == 'U') {
        // B = B * A
        for (unsigned int j = 0; j < n; j++) {
            mpfr_set(tmp, *alpha, rnd); // alpha * 1 for unitary matrix

            for (unsigned int i = 0; i < m; i++) {
                mpfr_mul(b_dat[i * ldb + j], b_dat[i * ldb + j], tmp, rnd);            
            }
            for (unsigned int k = j + 1; k < n; k++) {
                if (!mpfr_zero_p(a_dat[k*lda + j])) {
                    mpfr_mul(tmp, *alpha, a_dat[k*lda + j], rnd);
                    for (unsigned int i = 0; i < m; i++) {
                        mpfr_mul(prod, tmp, b_dat[i*ldb + k], rnd);
                        mpfr_add(b_dat[i * ldb + j], b_dat[i * ldb + j], prod, rnd);
                    }
                }
            }
        }
    } else if (side == 'R' && uplo == 'L' && transa == 'T' && diag == 'U') {
        // B = alpha * B * A^T
        for (unsigned int i = 0; i < m; i++) {
            for (unsigned int j = 0; j < n; j++) {
                unsigned int col = n - j - 1;
                mpfr_set_zero(tmp, 0);
                for (unsigned int l = 0; l <= col; l++) {
                    if (l < col) {
                        mpfr_mul(prod, b_dat[i*ldb + l], a_dat[col*lda + l], rnd);
                    } else {
                        mpfr_set(prod, b_dat[i*ldb + l], rnd);
                    }
                    mpfr_mul(prod, prod, *alpha, rnd);
                    mpfr_add(tmp, tmp, prod, rnd);
                }
                mpfr_set(b_dat[i*ldb + col], tmp, rnd);
            }
        }
    } else if (side == 'R' && uplo == 'U' && transa == 'N' && diag == 'N') {
        // B = alpha * B * A
        for (unsigned int i = 0; i < m; i++) {
            for (unsigned int j = 0; j < n; j++) {
                unsigned int col = n - j - 1;
                mpfr_set_zero(tmp, 0);
                for (unsigned int l = 0; l <= col; l++) {
                    mpfr_mul(prod, b_dat[i*ldb + l], a_dat[l*lda + col], rnd);
                    mpfr_mul(prod, prod, *alpha, rnd);
                    mpfr_add(tmp, tmp, prod, rnd);
                }
                mpfr_set(b_dat[i*ldb + col], tmp, rnd);
            }
        }
    }
}

}