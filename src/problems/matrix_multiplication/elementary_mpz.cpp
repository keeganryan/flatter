#include "elementary_mpz.h"

#include <cassert>

namespace flatter {
namespace MatrixMultiplicationImpl {

const std::string ElementaryMPZ::impl_name() {return "ElementaryMPZ";}

ElementaryMPZ::ElementaryMPZ(const Matrix& C, const Matrix& A, const Matrix& B,
                       bool accumulate_c,
                       const ComputationContext& cc) :
    Base(C, A, B, accumulate_c, cc)
{
    _is_configured = false;
    configure(C, A, B, accumulate_c, cc);
}

ElementaryMPZ::~ElementaryMPZ() {
    unconfigure();
}

void ElementaryMPZ::unconfigure() {
    assert(_is_configured);

    delete wsb;
    _is_configured = false;
}

void ElementaryMPZ::configure(const Matrix& C, const Matrix& A, const Matrix& B,
                           bool accumulate_c,
                           const ComputationContext& cc) {
    assert(C.type() == ElementType::MPZ);
    assert(A.type() == ElementType::MPZ);
    assert(B.type() == ElementType::MPZ);
    assert(!cc.is_threaded());
    
    if (_is_configured) {
        //unconfigure();
        Base::configure(C, A, B, accumulate_c, cc);
        wsb->set_precision(prec);
        return;
    }

    if (!C.is_transposed()) {
        Base::configure(C, A, B, accumulate_c, cc);
    } else {
        Base::configure(C.transpose(), B.transpose(), A.transpose(), accumulate_c, cc);
    }

    dA = this->A.data<mpz_t>();
    dB = this->B.data<mpz_t>();
    dC = this->C.data<mpz_t>();

    unsigned int sz_needed = 2;
    wsb = new WorkspaceBuffer<mpz_t>(sz_needed, prec);
    _is_configured = true;
}

void ElementaryMPZ::solve() {
    log_start();

    gemm();

    log_end();
}

void ElementaryMPZ::gemm() {
    unsigned int lda = dA.stride();
    unsigned int ldb = dB.stride();

    if (!dA.is_transposed() && !dB.is_transposed()) {
        gemm_xx(lda, 1, ldb, 1);
    } else if (!dA.is_transposed() && dB.is_transposed()) {
        gemm_xx(lda, 1, 1, ldb);
    } else if (dA.is_transposed() && !dB.is_transposed()) {
        gemm_xx(1, lda, ldb, 1);  
    } else {
        gemm_xx(1, lda, 1, ldb);
    }
}

void ElementaryMPZ::gemm_xx(unsigned int adr, unsigned int adc, unsigned int bdr, unsigned int bdc) {
    mpz_t* work = wsb->walloc(2);
    mpz_t& prod = work[0];
    mpz_t& sum = work[1];

    mpz_t* A = this->dA.get_data();
    mpz_t* B = this->dB.get_data();
    mpz_t* C = this->dC.get_data();
    unsigned int ldc = this->dC.stride();

    // C = alpha * A^T . B + beta * C
    for (unsigned int i = 0; i < m; i += 1) {
        for (unsigned int j = 0; j < n; j += 1) {
            mpz_set_ui(sum, 0);
            for (unsigned int l = 0; l < k; l += 1) {
                mpz_mul(prod, A[i*adr + l*adc], B[l*bdr + j*bdc]);
                mpz_add(sum, sum, prod);
            }
            if (_accumulate_C) {
                mpz_add(C[i*ldc + j], C[i*ldc + j], sum);
            } else {
                mpz_set(C[i*ldc + j], sum);
            }
        }
    }

    wsb->wfree(work, 2);
}

}
}