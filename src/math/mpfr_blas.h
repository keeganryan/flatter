#pragma once

#include <mpfr.h>

void copy(unsigned int n, mpfr_t* x, unsigned int incx, mpfr_t* y, unsigned int incy);

// Do C = alpha * A * B + beta * C where A and B may be transposed
void gemm(char transa, char transb, unsigned int m, unsigned int n, unsigned int k,
          const mpfr_t& alpha, mpfr_t* A, unsigned int lda, mpfr_t* B, unsigned int ldb,
          const mpfr_t& beta, mpfr_t* C, unsigned int ldc, mpfr_t* work, unsigned int lwork);

// Do y = alpha * A*x + beta * y or y = alpha * A^T * x + beta * y
void gemv(char trans, unsigned int m, unsigned int n, const mpfr_t& alpha, mpfr_t* A, unsigned int lda,
          mpfr_t* x, unsigned int incx, const mpfr_t& beta, mpfr_t* y, unsigned int incy, mpfr_t* work, unsigned int lwork);

// Do A += alpha * x * y^T
void ger(unsigned int m, unsigned int n, const mpfr_t& alpha, mpfr_t* x, unsigned int incx,
         mpfr_t* y, unsigned int incy, mpfr_t* A, unsigned int lda, mpfr_t* work, unsigned int lwork);

void trmm(char side, char uplo, char transa, char diag,
          unsigned int m, unsigned int n, const mpfr_t& alpha, mpfr_t* A, unsigned int lda,
          mpfr_t* B, unsigned int ldb, mpfr_t* work, unsigned int lwork);

// Do x = A*x where  A is triangular
void trmv(char uplo, char trans, char diag, unsigned int n, mpfr_t* A, unsigned int lda,
          mpfr_t* x, unsigned int incx, mpfr_t* work, unsigned int lwork);