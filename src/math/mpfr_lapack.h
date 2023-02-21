#pragma once

#include <mpfr.h>

void geqr2(unsigned int m, unsigned int n, mpfr_t* A, unsigned int lda,
           mpfr_t* tau, mpfr_t* work, unsigned int lwork);

void geqrt(unsigned int m, unsigned int n, unsigned int nb,
           mpfr_t* A, unsigned int lda,
           mpfr_t* T, unsigned int ldt, mpfr_t* work, unsigned int lwork);

void geqrt2(unsigned int m, unsigned int n, mpfr_t* A, unsigned int lda,
            mpfr_t* T, unsigned int ldt, mpfr_t* work, unsigned int lwork);

void larf(unsigned int m, unsigned int n,
            mpfr_t* v, unsigned int incv,
            const mpfr_t& tau,
            mpfr_t* C, unsigned int ldc,
            mpfr_t* work, unsigned int lwork);

void larfb(char side, char trans, char direct, char storev,
           unsigned int m, unsigned int n, unsigned int k,
           mpfr_t* V, unsigned int ldv,
           mpfr_t* T, unsigned int ldt,
           mpfr_t* C, unsigned int ldc,
           mpfr_t* work, unsigned int lwork);

void larfg(unsigned int n, mpfr_t& alpha, mpfr_t* x, unsigned int incx,
           mpfr_t& tau, mpfr_t* work, unsigned int lwork);

void latsqr(unsigned int m, unsigned int n, unsigned int mb, unsigned int nb,
            mpfr_t* A, unsigned int lda,
            mpfr_t* T, unsigned int ldt,
            mpfr_t* work, unsigned int lwork);