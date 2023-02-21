#include "generic.h"

#include "problems/qr_factorization.h"
#include "problems/relative_size_reduction.h"

#include <cassert>

namespace flatter {
namespace RelativeSizeReductionImpl {

const std::string Generic::impl_name() {return "Generic";}

Generic::Generic(const RelativeSizeReductionParams& params, const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

Generic::~Generic() {
    if (_is_configured) {
        unconfigure();
    }
}

void Generic::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Generic::configure(const RelativeSizeReductionParams& params, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    
    Base::configure(params, cc);

    _is_configured = true;
}

void Generic::solve() {
    log_start();

    unsigned int prec = params.B1.prec();
    Matrix tau (ElementType::MPFR, params.B1.ncols(), 1, prec);
    Matrix RV (ElementType::MPFR, params.B1.nrows(), params.B1.ncols(), prec);
    Matrix R2 (ElementType::MPFR, params.B2.nrows(), params.B2.ncols(), prec);

    Matrix::copy(RV, params.B1);

    QRFactorization qr(RV, tau, cc);
    qr.solve();

    RelativeSizeReductionParams new_params (params.B1, RV, tau, params.B2, params.U);
    new_params.R2 = R2;
    RelativeSizeReduction rsr(new_params, cc);
    rsr.solve();

    log_end();
}

}
}