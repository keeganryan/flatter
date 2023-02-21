#pragma once

#include "qr_factorization/base.h"

namespace flatter {

class QRFactorization : public QRFactorizationImpl::Base {
public:
    QRFactorization();
    QRFactorization(const Matrix& A, const Matrix& tau, const Matrix& T,
                    const ComputationContext& cc);
    QRFactorization(const Matrix& A, const Matrix& tau,
                    const ComputationContext& cc);
    QRFactorization(const Matrix& A,
                    const ComputationContext& cc);

    ~QRFactorization();

    void configure(const Matrix& A, const Matrix& tau, const Matrix& T, const ComputationContext& cc);
    void configure(const Matrix& A, const Matrix& tau, const ComputationContext& cc);
    void configure(const Matrix& A, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool _is_configured;
    QRFactorizationImpl::Base* qr;
};

}