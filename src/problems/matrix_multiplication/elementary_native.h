#pragma once

#include <gmp.h>

#include "data/matrix.h"
#include "workspace_buffer.h"

#include "problems/matrix_multiplication/base.h"

namespace flatter {
namespace MatrixMultiplicationImpl {

template <class T, class U, class V>
class Elementary : public Base {
public:
    Elementary(const Matrix& C, const Matrix& A, const Matrix& B,
               bool accumulate_c,
               const ComputationContext& cc);
    ~Elementary();

    const std::string impl_name();

    void configure(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc);
    void solve();

private:
    void unconfigure();

    void gemm();
    void gemm_xx(unsigned int adr, unsigned int adc, unsigned int bdr, unsigned int bdc);

    MatrixData<T> dC;
    MatrixData<U> dA;
    MatrixData<V> dB;

    bool _is_configured;
};

}
}