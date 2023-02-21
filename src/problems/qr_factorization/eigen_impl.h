#pragma once

#include "problems/qr_factorization/base.h"
#include "workspace_buffer.h"

namespace flatter {
namespace QRFactorizationImpl {

class Eigen : public Base {
public:
    Eigen(const Matrix& A, const Matrix& tau, const Matrix& T,
                const ComputationContext& cc);
    ~Eigen();

    const std::string impl_name();

    void configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                   const ComputationContext& cc);
    void solve();

private:
    void unconfigure();
    void clear_subdiagonal();

    bool _is_configured;
    bool _save_tau;
    bool _save_block_reflector;

    MatrixData<double> dA;
    MatrixData<double> dT;
    double* tau_ptr;
};

}
}