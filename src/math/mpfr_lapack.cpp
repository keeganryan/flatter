/*
Parts of this code are derived from the LAPACK linear algebra library, release
under the modified BSD license.

Copyright (c) 1992-2022 The University of Tennessee and The University
                        of Tennessee Research Foundation.  All rights
                        reserved.
Copyright (c) 2000-2022 The University of California Berkeley. All
                        rights reserved.
Copyright (c) 2006-2022 The University of Colorado Denver.  All rights
                        reserved.
Copyright (c) 2023      Keegan Ryan

$COPYRIGHT$

Additional copyrights may follow

$HEADER$

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

- Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

- Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer listed
  in this license in the documentation and/or other materials
  provided with the distribution.

- Neither the name of the copyright holders nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

The copyright holders provide no reassurances that the source code
provided does not infringe any patent, copyright, or any other
intellectual property rights of third parties.  The copyright holders
disclaim any liability to any recipient for claims brought against
recipient by any third party for infringement of that parties
intellectual property rights.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "mpfr_lapack.h"

#include <algorithm>
#include <cassert>

#include "mpfr_blas.h"
#include "workspace_buffer.h"
#include "data/matrix/matrix_data.h"
#include "problems/matrix_multiplication/tri_matmul.h"
#include "problems/matrix_multiplication.h"

static mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

void geqr2(unsigned int m, unsigned int n, mpfr_t* A, unsigned int lda,
           mpfr_t* tau, mpfr_t* work, unsigned int lwork) {
    unsigned int n_reflectors = std::min(m, n) - 1;
    for (unsigned int i = 0; i < n_reflectors; i++) {
        larfg(m - i, A[i*lda+i], &A[(i+1) * lda + i], lda, tau[i], work, lwork);
        larf(m-i, n-i-1, &A[i * lda + i], lda, tau[i], &A[i * lda + i + 1], lda, work, lwork);
    }
}

void geqrt(unsigned int m, unsigned int n, unsigned int nb, mpfr_t* A, unsigned int lda,
           mpfr_t* T, unsigned int ldt, mpfr_t* work, unsigned int lwork) {
    unsigned int k = std::min(m, n);
    unsigned int bs = nb;

    for (unsigned int i = 0; i < k; i += bs) {
        unsigned int ib = std::min(bs, k - i);
        geqrt2(m - i, ib, &A[i * lda + i], lda, &T[i], ldt, work, lwork);

        if (i + ib < k) {
            larfb('L', 'T', 'F', 'C', m - i, n - (i + ib), ib, &A[i * lda + i], lda, &T[i], ldt, &A[i * lda + i + ib], lda, work, lwork); //n - (i + ib));
        }
    }
}

void geqrt2(unsigned int m, unsigned int n, mpfr_t* A, unsigned int lda,
            mpfr_t* T, unsigned int ldt, mpfr_t* work, unsigned int lwork) {
    assert(lwork >= 4);
    mpfr_t& zero = work[0];
    mpfr_t& one = work[1];
    mpfr_t& aii = work[2];
    mpfr_t& alpha = work[3];
    mpfr_set_zero(zero, 0);
    mpfr_set_ui(one, 1, rnd);
    work += 4;
    lwork -= 4;

    unsigned int k = std::min(m, n);
    for (unsigned int i = 0; i < k; i++) {
        mpfr_t* subA = &A[i * lda + i + 1];
        mpfr_t* v = &A[i * lda + i];
        unsigned int incv = lda;

        larfg(m - i, v[0], v+incv, incv, T[i * ldt], work, lwork);
        if (i < k - 1) {
            mpfr_set(aii, A[i*lda + i], rnd);
            mpfr_set_ui(A[i*lda + i], 1, rnd);
            gemv('T', m-i, n-i-1, one, subA, lda, v, incv, zero, &T[n - 1], ldt, work, lwork);

            mpfr_neg(alpha, T[i * ldt], rnd);
            ger(m-i, n-i-1, alpha, v, incv, &T[n - 1], ldt, subA, lda, work, lwork);

            mpfr_set(A[i*lda + i], aii, rnd);
        }
    }

    for (unsigned int i = 1; i < k; i++) {
        mpfr_set(aii, A[i*lda + i], rnd);
        mpfr_set_ui(A[i*lda + i], 1, rnd);

        mpfr_neg(alpha, T[i * ldt], rnd);
        gemv('T', m-i, i, alpha, &A[i*lda], lda, &A[i*lda + i], lda, zero, &T[i], ldt, work, lwork);
        mpfr_set(A[i*lda + i], aii, rnd);

        trmv('U', 'N', 'N', i, T, ldt, &T[i], ldt, work, lwork);

        mpfr_set(T[i*ldt + i], T[i*ldt], rnd);
        mpfr_set_zero(T[i*ldt], 0);
    }
}

void larf(unsigned int m, unsigned int n,
          mpfr_t* v, unsigned int incv,
          const mpfr_t& tau,
          mpfr_t* C, unsigned int ldc,
          mpfr_t* work, unsigned int lwork) {
    if (m <= 1) {
        return;
    }
    assert(lwork >= 2);

    // (I - tau * v * vT) * C
    mpfr_t& vtC = work[0];
    mpfr_t& tmp = work[1];

    for (unsigned int j = 0; j < n; j++) {
        mpfr_set(vtC, C[j], rnd);
        for (unsigned int i = 1; i < m; i++) {
            mpfr_mul(tmp, v[incv*i], C[ldc*i + j], rnd);
            mpfr_add(vtC, vtC, tmp, rnd);
        }
        mpfr_mul(vtC, vtC, tau, rnd);

        mpfr_sub(C[j], C[j], vtC, rnd);
        for (unsigned int i = 1; i < m; i++) {
            mpfr_mul(tmp, vtC, v[incv * i], rnd);
            mpfr_sub(C[ldc*i+j], C[ldc*i+j], tmp, rnd);
        }
    }
}

void larfb(char side, char trans, char direct, char storev,
           unsigned int m, unsigned int n, unsigned int k,
           mpfr_t* V, unsigned int ldv,
           mpfr_t* T, unsigned int ldt,
           mpfr_t* C, unsigned int ldc,
           mpfr_t* work, unsigned int lwork) {
    assert(side == 'L' || side == 'R');
    assert(trans == 'N' || trans == 'T');
    assert(direct == 'F' || direct == 'B');
    assert(storev == 'C' || storev == 'R');

    // Only handle this case for now
    assert(side == 'L' && trans == 'T' && direct == 'F' && storev == 'C');
    // Compute C = (I - V.T.V^T)^T . C
    // C = C - V.(T^T.(V^T.C))

    flatter::WorkspaceBuffer<mpfr_t> wsb(12, mpfr_get_prec(work[0]));

    char trans_t = (trans == 'N') ? 'T' : 'N';
    // C is mxn
    // V is mxk
    // W is kxn
    assert(lwork >= k*n + 2);
    mpfr_t& one = work[0];
    mpfr_t& neg_one = work[1];
    mpfr_t* W = &work[2];
    unsigned int ldw = k;
    work += k*n + 2;
    lwork -= k*n + 2;
    mpfr_set_ui(one, 1, rnd);
    mpfr_set_si(neg_one, -1, rnd);

    // W = C1^T
    for (unsigned int i = 0; i < k; i++) {
        copy(n, &C[i*ldc], 1, &W[i], ldw);
    }
    // W = W * V1

    flatter::MatrixData<mpfr_t> W_mat(W, n, k, ldw);
    flatter::MatrixData<mpfr_t> V1_mat(V, k, k, ldv);
    flatter::ComputationContext cc;

    flatter::TriMatrixMultiplication tmm1(V1_mat, W_mat, 'R', 'L', 'N', 'U', &one, &wsb, cc);
    tmm1.solve();

    if (m > k) {
        // W += C2^T * V2
        flatter::MatrixData<mpfr_t> C_mat(C, m, n, ldc);
        flatter::MatrixData<mpfr_t> C2T = C_mat.submatrix(k,m,0,n).transpose();
        flatter::MatrixData<mpfr_t> V2_mat(&V[k*ldv], m-k, k, ldv);
        flatter::MatrixMultiplication mm(W_mat, C2T, V2_mat, true, cc);
        mm.solve();
    }

    // W = W * T^T
    flatter::MatrixData<mpfr_t> T_mat(T, k, k, ldt);
    flatter::TriMatrixMultiplication tmm2(T_mat, W_mat, 'R', 'U', trans_t, 'N', &one, &wsb, cc);
    tmm2.solve();

    if (m > k) {
        // C2 -= V2 * W^T
        flatter::MatrixData<mpfr_t> C_mat(C, m, n, ldc);
        flatter::MatrixData<mpfr_t> C2 = C_mat.submatrix(k, m, 0, n);
        flatter::MatrixData<mpfr_t> V2_mat(&V[k*ldv], m-k, k, ldv);
        flatter::MatrixData<mpfr_t> WT = W_mat.transpose();
        // TODO THIS IS INCORRECT, we should do C2 -= V2_mat * W^T, not +=
        flatter::MatrixMultiplication mm2(C2, V2_mat, WT, false, cc);

        mm2.solve();
    }

    // W = W * V1^T
    flatter::TriMatrixMultiplication tmm3(V1_mat, W_mat, 'R', 'L', 'T', 'U', &one, &wsb, cc);
    tmm3.solve();
    // C1 -= W^T
    for (unsigned int i = 0; i < k; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mpfr_sub(C[i*ldc + j], C[i*ldc + j], W[j*ldw + i], rnd);
        }
    }

}

void larfg(unsigned int n, mpfr_t& alpha, mpfr_t* x, unsigned int incx, mpfr_t& tau, mpfr_t* work, unsigned int lwork) {
    assert(lwork >= 4);
    mpfr_t& sigma = work[0];
    mpfr_t& v1 = work[1];
    mpfr_t& mu = work[2];
    mpfr_t& tmp = work[3];

    mpfr_set_zero(sigma, 0);
    for (unsigned int i = 1; i < n; i++) {
        mpfr_sqr(tmp, x[incx*(i-1)], rnd);
        mpfr_add(sigma, sigma, tmp, rnd);
    }

    if (mpfr_zero_p(sigma)) {
        mpfr_set_zero(tau, 0);
        return;
    }

    mpfr_set(v1, alpha, rnd);
    mpfr_sqr(tmp, v1, rnd);
    mpfr_add(tmp, tmp, sigma, rnd);
    mpfr_sqrt(mu, tmp, rnd);
    if (mpfr_sgn(v1) <= 0) {
        mpfr_sub(v1, v1, mu, rnd);
    } else {
        mpfr_add(tmp, v1, mu, rnd);
        mpfr_div(v1, sigma, tmp, rnd);
        mpfr_neg(v1, v1, rnd);
    }

    // tau = 2 * v1 * v1 / (sigma + v1 * v1);
    mpfr_sqr(tau, v1, rnd);
    mpfr_add(tmp, sigma, tau, rnd);
    mpfr_div(tau, tau, tmp, rnd);
    mpfr_mul_2ui(tau, tau, 1, rnd);

    mpfr_set(alpha, mu, rnd);
    for(unsigned int i = 1; i < n; i++) {
        mpfr_div(x[(i-1)*incx], x[(i-1)*incx], v1, rnd);
    }
}


void latsqr(unsigned int m, unsigned int n, unsigned int mb, unsigned int nb,
            mpfr_t* A, unsigned int lda,
            mpfr_t* T, unsigned int ldt,
            mpfr_t* work, unsigned int lwork) {
    if (m == 0 || n == 0) {
        return;
    }

    geqrt(m, n, nb, A, lda, T, ldt, work, lwork);
}