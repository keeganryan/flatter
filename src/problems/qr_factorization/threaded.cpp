#include "threaded.h"

#include "problems/qr_factorization.h"

#include <cassert>

namespace flatter {
namespace QRFactorizationImpl {

const std::string Threaded::impl_name() {return "Threaded";}

Threaded::Threaded(const Matrix& A, const Matrix& tau, const Matrix& T,
        const ComputationContext& cc) :
    Base(A, tau, T, cc)
{
    _is_configured = false;
    configure(A, tau, T, cc);
}

Threaded::~Threaded() {
    if (_is_configured) {
        unconfigure();
    }
}

void Threaded::unconfigure() {
    assert(_is_configured);
    _is_configured = false;
}

void Threaded::configure(const Matrix& A, const Matrix& tau, const Matrix& T, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    assert(cc.is_threaded());

    Base::configure(A, tau, T, cc);

    _is_configured = true;
}

void Threaded::solve() {
    log_start();

    ComputationContext sub_cc(1);

    #pragma omp single
    {
        QRFactorization qr(A, tau, T, sub_cc);
        qr.solve();
    }

    log_end();
}

}
}