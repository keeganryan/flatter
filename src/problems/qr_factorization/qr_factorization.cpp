#include "problems/qr_factorization.h"

#include <cassert>

#include "blocked.h"
#include "householder_mpfr.h"
#include "eigen_impl.h"
#include "threaded.h"

namespace flatter {

QRFactorization::QRFactorization() {
    _is_configured = false;
}

QRFactorization::QRFactorization(const Matrix& A,
                                 const ComputationContext& cc) :
    QRFactorization(A, Matrix(), Matrix(), cc)
{}

QRFactorization::QRFactorization(const Matrix& A, const Matrix& tau,
                                 const ComputationContext& cc) :
    QRFactorization(A, tau, Matrix(), cc)
{}

QRFactorization::QRFactorization(const Matrix& A, const Matrix& tau, const Matrix& T,
                                 const ComputationContext& cc) :
    Base(A, tau, T, cc)
{
    _is_configured = false;
    configure(A, tau, T, cc);
}

QRFactorization::~QRFactorization() {
    if (_is_configured) {
        unconfigure();
    }
}

void QRFactorization::unconfigure() {
    assert(_is_configured);
    delete this->qr;
    _is_configured = false;
}
void QRFactorization::configure(const Matrix& A,
                                const ComputationContext& cc) {
    configure(A, Matrix(), Matrix(), cc);
}
void QRFactorization::configure(const Matrix& A, const Matrix& tau,
                                const ComputationContext& cc) {
    configure(A, tau, Matrix(), cc);
}
void QRFactorization::configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                                const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    Base::configure(A, tau, T, cc);
    if (this->rank >= 1000 && false) {
        this->qr = new QRFactorizationImpl::Blocked(A, tau, T, cc);
    } else {
        if (A.type() == ElementType::MPFR) {
            this->qr = new QRFactorizationImpl::HouseholderMPFR(A, tau, T, cc);
        } else if (A.type() == ElementType::DOUBLE) {
            this->qr = new QRFactorizationImpl::Eigen(A, tau, T, cc);
        } else {
            assert(0);
        }
    }

    _is_configured = true;
}

void QRFactorization::solve() {
    assert(_is_configured);
    bool use_tasks = cc.is_threaded();

    if (use_tasks && omp_get_active_level() == 0) {
        #pragma omp parallel num_threads(cc.nthreads()) if (use_tasks)
        {
            #pragma omp single
            {
                qr->solve();
            }
        }
    } else {
        #pragma omp taskgroup
        {
            qr->solve();
        }
    }
}

}
