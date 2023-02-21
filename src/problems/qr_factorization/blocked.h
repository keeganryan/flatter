#pragma once

#include "workspace_buffer.h"
#include "problems/qr_factorization.h"

namespace flatter {
namespace QRFactorizationImpl {

class Blocked : public Base {
public:
    Blocked(const Matrix& A, const Matrix& tau, const Matrix& T,
                const ComputationContext& cc);
    ~Blocked();

    const std::string impl_name();

    void configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                   const ComputationContext& cc);
    void solve();

private:
    void unconfigure();

    bool _is_configured;
    bool _save_tau;
    bool _save_block_reflector;

    unsigned int r0;
    unsigned int r1;
    MatrixData<mpfr_t> dA;
    MatrixData<mpfr_t> dT;
    mpfr_t* tau_ptr;
    MatrixData<mpfr_t> T_sub_0;
    MatrixData<mpfr_t> T_sub_1;

    QRFactorization qr0;
    QRFactorization qr1;

    WorkspaceBuffer<mpfr_t>* wsb;
    unsigned int alloc_size;
    unsigned int lwork;
    mpfr_t* work;
};

}
}