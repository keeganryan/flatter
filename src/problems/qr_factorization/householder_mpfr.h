#pragma once

#include "problems/qr_factorization/base.h"
#include "workspace_buffer.h"

namespace flatter {
namespace QRFactorizationImpl {

class HouseholderMPFR : public Base {
public:
    HouseholderMPFR(const Matrix& A, const Matrix& tau, const Matrix& T,
                const ComputationContext& cc);
    ~HouseholderMPFR();

    const std::string impl_name();

    void configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                   const ComputationContext& cc);
    void solve();

private:
    void unconfigure();

    void generate_householder(unsigned int i);
    void apply_householder(unsigned int i);
    void clear_subdiagonal();

    bool _is_configured;
    bool _save_tau;
    bool _save_block_reflector;
    MatrixData<mpfr_t> dA;
    MatrixData<mpfr_t> dT;
    mpfr_t* tau_ptr;
    WorkspaceBuffer<mpfr_t>* wsb;
    unsigned int alloc_size;
    unsigned int lwork;
    mpfr_t* work;
    mpfr_t* p_ZERO;
};

}
}