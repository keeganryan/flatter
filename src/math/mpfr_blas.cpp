#include "mpfr_blas.h"

#include <cassert>

#include "problems/matrix_multiplication/tri_matmul.h"
#include "workspace_buffer.h"

static mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

void copy(unsigned int n, mpfr_t* x, unsigned int incx, mpfr_t* y, unsigned int incy) {
    for (unsigned int i = 0; i < n; i++) {
        mpfr_set(y[i * incy], x[i * incx], rnd);
    }
}

void gemm_nn(unsigned int m, unsigned int n, unsigned int k,
          const mpfr_t& alpha, mpfr_t* A, unsigned int lda, mpfr_t* B, unsigned int ldb,
          const mpfr_t& beta, mpfr_t* C, unsigned int ldc, mpfr_t* work, unsigned int lwork) {
    assert(lwork >= 2);
    mpfr_t& prod = work[0];
    mpfr_t& sum = work[1];

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if (mpfr_zero_p(beta)) {
                mpfr_set_zero(C[i * ldc + j], 0);
            } else {
                mpfr_mul(C[i*ldc + j], C[i*ldc + j], beta, rnd);
            }
        }
    }

    // C = alpha * A^T . B + beta * C
    for (unsigned int i = 0; i < m; i += 1) {
        for (unsigned int j = 0; j < n; j += 1) {
            mpfr_set_zero(sum, 0);
            for (unsigned int l = 0; l < k; l += 1) {
                mpfr_mul(prod, A[i*lda + l], B[l*ldb + j], rnd);
                mpfr_add(sum, sum, prod, rnd);
            }
            mpfr_mul(sum, sum, alpha, rnd);
            mpfr_add(C[i*ldc + j], C[i*ldc + j], sum, rnd);
        }
    }
}

void gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
          const mpfr_t& alpha, mpfr_t* A, unsigned int lda, mpfr_t* B, unsigned int ldb,
          const mpfr_t& beta, mpfr_t* C, unsigned int ldc, mpfr_t* work, unsigned int lwork) {
    assert(transa == 'N' || transa == 'T');
    assert(transb == 'N' || transb == 'T');

    assert(lwork >= 2);
    mpfr_t& prod = work[0];
    mpfr_t& sum = work[1];

    assert(
        (transa == 'N' && transb == 'N') ||
        (transa == 'T' && transb == 'N') ||
        (transa == 'N' && transb == 'T')
    );

    if (transa == 'N' && transb == 'N') {
        gemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, work, lwork);
    } else if (transa == 'T' && transb == 'N') {
        // C = alpha * A^T . B + beta * C
        for (unsigned int j = 0; j < n; j++) {
            for (unsigned int i = 0; i < m; i++) {
                mpfr_set_zero(sum, 0);
                for (unsigned int l = 0; l < k; l++) {
                    mpfr_mul(prod, A[l*lda + i], B[l*ldb + j], rnd);
                    mpfr_add(sum, sum, prod, rnd);
                }
                mpfr_mul(sum, sum, alpha, rnd);
                mpfr_mul(C[i*ldc + j], C[i*ldc + j], beta, rnd);
                mpfr_add(C[i*ldc + j], C[i*ldc + j], sum, rnd);
            }
        }
    } else if (transa == 'N' && transb == 'T') {
        // C = alpha * A . B^T + beta * C
        for (unsigned int j = 0; j < n; j++) {
            for (unsigned int i = 0; i < m; i++) {
                mpfr_set_zero(sum, 0);
                for (unsigned int l = 0; l < k; l++) {
                    mpfr_mul(prod, A[i*lda + l], B[j*ldb + l], rnd);
                    mpfr_add(sum, sum, prod, rnd);
                }
                mpfr_mul(sum, sum, alpha, rnd);
                mpfr_mul(C[i*ldc + j], C[i*ldc + j], beta, rnd);
                mpfr_add(C[i*ldc + j], C[i*ldc + j], sum, rnd);
            }
        }  
    }
}

void gemv(char trans, unsigned int m, unsigned int n, const mpfr_t& alpha, mpfr_t* A, unsigned int lda,
          mpfr_t* x, unsigned int incx, const mpfr_t& beta, mpfr_t* y, unsigned int incy,
          mpfr_t* work, unsigned int lwork) {
    assert(lwork > 2);
    assert(trans == 'N' || trans == 'T');

    mpfr_t& prod = work[0];
    mpfr_t& sum = work[1];

    if (trans == 'N') {
         // y = A * x
        for (unsigned int i = 0; i < m; i++) {
            mpfr_set_zero(sum, 0);
            for (unsigned int j = 0; j < n; j++) {
                mpfr_mul(prod, A[i*lda + j], x[j * incx], rnd);
                mpfr_add(sum, sum, prod, rnd);
            }
            mpfr_mul(sum, sum, alpha, rnd);
            if (mpfr_zero_p(beta)) {
                mpfr_set(y[incy * i], sum, rnd);
            } else {
                mpfr_mul(y[incy * i], y[incy * i], beta, rnd);
                mpfr_add(y[incy * i], y[incy * i], sum, rnd);
            }
        }       
    } else {
        // y = A^T * x
        for (unsigned int i = 0; i < n; i++) {
            mpfr_set_zero(sum, 0);
            for (unsigned int j = 0; j < m; j++) {
                mpfr_mul(prod, A[j*lda + i], x[j * incx], rnd);
                mpfr_add(sum, sum, prod, rnd);
            }
            mpfr_mul(sum, sum, alpha, rnd);
            if (mpfr_zero_p(beta)) {
                mpfr_set(y[incy * i], sum, rnd);
            } else {
                mpfr_mul(y[incy * i], y[incy * i], beta, rnd);
                mpfr_add(y[incy * i], y[incy * i], sum, rnd);
            }
        }
    }
}

void ger(unsigned int m, unsigned int n, const mpfr_t& alpha, mpfr_t* x, unsigned int incx,
         mpfr_t* y, unsigned int incy, mpfr_t* A, unsigned int lda, mpfr_t* work, unsigned int lwork) {
    assert(lwork >= 1);
    mpfr_t& prod = work[0];
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mpfr_mul(prod, x[i * incx], y[j * incy], rnd);
            mpfr_mul(prod, prod, alpha, rnd);
            mpfr_add(A[i*lda + j], A[i*lda + j], prod, rnd);
        }
    }
}

void trmv(char uplo, char trans, char diag, unsigned int n, mpfr_t* A, unsigned int lda,
          mpfr_t* x, unsigned int incx, mpfr_t* work, unsigned int lwork) {
    assert(uplo == 'U' || uplo == 'L');
    assert(trans == 'N' || trans == 'T');
    assert(diag == 'U' || diag == 'N');

    assert(lwork >= 1);
    mpfr_t& tmp = work[0];

    assert(uplo == 'U' && trans == 'N' && diag == 'N');
    // x = Ax, A upper triangular, not unit triangular
    for (unsigned int i = 0; i < n; i++) {
        mpfr_mul(x[i*incx], x[i*incx], A[i*lda + i], rnd);
        for (unsigned int j = i + 1; j < n; j++) {
            mpfr_mul(tmp, x[j * incx], A[i*lda + j], rnd);
            mpfr_add(x[i*incx], x[i*incx], tmp, rnd);
        }
    }
}

void trmm(char side, char uplo, char transa, char diag,
          unsigned int m, unsigned int n, const mpfr_t& alpha, mpfr_t* A, unsigned int lda,
          mpfr_t* B, unsigned int ldb, mpfr_t* work, unsigned int lwork) {
    flatter::WorkspaceBuffer<mpfr_t> wsb(2, mpfr_get_prec(work[0]));

    int a_size = (side == 'R') ? n : m;
    flatter::MatrixData<mpfr_t> mat_A(A, a_size, a_size, lda);
    flatter::MatrixData<mpfr_t> mat_B(B, m, n, ldb);
    flatter::ComputationContext cc;

    flatter::TriMatrixMultiplication tmm(mat_A, mat_B,
        side, uplo, transa, diag, &alpha, &wsb, cc);
    tmm.solve();
}