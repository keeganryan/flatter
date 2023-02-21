#include "elementary_native.h"

#include <cassert>

namespace flatter {
namespace MatrixMultiplicationImpl {

template <class T, class U, class V>
const std::string Elementary<T, U, V>::impl_name() {return "Elementary";}

template <class T, class U, class V>
Elementary<T, U, V>::Elementary(const Matrix& C, const Matrix& A, const Matrix& B,
                       bool accumulate_c,
                       const ComputationContext& cc) :
    Base(C, A, B, accumulate_c, cc)
{
    _is_configured = false;
    configure(C, A, B, accumulate_c, cc);
}

template <class T, class U, class V>
Elementary<T, U, V>::~Elementary() {
    unconfigure();
}

template <class T, class U, class V>
void Elementary<T, U, V>::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

template <class T, class U, class V>
void Elementary<T, U, V>::configure(const Matrix& C, const Matrix& A, const Matrix& B,
                           bool accumulate_c,
                           const ComputationContext& cc) {
    assert(Matrix::is_type<T>(C));
    assert(Matrix::is_type<U>(A));
    assert(Matrix::is_type<V>(B));
    assert(!cc.is_threaded());
    
    if (_is_configured) {
        //unconfigure();
        Base::configure(C, A, B, accumulate_c, cc);
        return;
    }

    if (!C.is_transposed()) {
        Base::configure(C, A, B, accumulate_c, cc);
    } else {
        Base::configure(C.transpose(), B.transpose(), A.transpose(), accumulate_c, cc);
    }

    dC = this->C.template data<T>();
    dA = this->A.template data<U>();
    dB = this->B.template data<V>();

    _is_configured = true;
}

template <class T, class U, class V>
void Elementary<T, U, V>::solve() {
    log_start();

    gemm();

    log_end();
}

template <class T, class U, class V>
void Elementary<T, U, V>::gemm() {
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

template <class T, class U, class V>
void Elementary<T, U, V>::gemm_xx(unsigned int adr, unsigned int adc, unsigned int bdr, unsigned int bdc) {
    T prod;
    T sum;

    T* C = this->dC.get_data();
    U* A = this->dA.get_data();
    V* B = this->dB.get_data();
    unsigned int ldc = this->dC.stride();

    // C = alpha * A^T . B + beta * C
    for (unsigned int i = 0; i < m; i += 1) {
        for (unsigned int j = 0; j < n; j += 1) {
            sum = 0;
            for (unsigned int l = 0; l < k; l += 1) {
                prod = A[i*adr + l*adc] * B[l*bdr + j*bdc];
                sum += prod;
            }
            if (_accumulate_C) {
                C[i*ldc + j] += sum;
            } else {
                C[i*ldc + j] = sum;
            }
        }
    }
}

template class Elementary<int64_t, int64_t, int64_t>;
template class Elementary<double, double, double>;
template class Elementary<double, double, int64_t>;

}
}