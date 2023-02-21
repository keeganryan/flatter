#pragma once

#include <gmp.h>

#include "data/matrix.h"
#include "workspace_buffer.h"

#include "problems/matrix_multiplication/base.h"

namespace flatter {
namespace MatrixMultiplicationImpl {

class ElementaryMPZ : public Base {
public:
    ElementaryMPZ(const Matrix& C, const Matrix& A, const Matrix& B,
               bool accumulate_c,
               const ComputationContext& cc);
    ~ElementaryMPZ();

    const std::string impl_name();

    void configure(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc);
    void solve();

private:
    void unconfigure();

    void gemm();
    void gemm_xx(unsigned int adr, unsigned int adc, unsigned int bdr, unsigned int bdc);

    MatrixData<mpz_t> dC;
    MatrixData<mpz_t> dA;
    MatrixData<mpz_t> dB;

    bool _is_configured;
    WorkspaceBuffer<mpz_t>* wsb;
};

}
}