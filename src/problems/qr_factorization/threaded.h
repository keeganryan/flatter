#pragma once

#include "workspace_buffer.h"

#include "problems/qr_factorization/base.h"

namespace flatter {
namespace QRFactorizationImpl {

class Threaded : public Base {
public:
    Threaded(const Matrix& A, const Matrix& tau, const Matrix& T,
                const ComputationContext& cc);
    ~Threaded();

    const std::string impl_name();

    void configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                const ComputationContext& cc);
    void solve();

private:
    void unconfigure();

    bool _is_configured;
};

}
}