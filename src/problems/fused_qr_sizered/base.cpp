#include "problems/fused_qr_sizered/base.h"

#include <cassert>
#include <sstream>

namespace flatter {
namespace FusedQRSizeRedImpl {

const std::string Base::prob_name() {return "Fused QR Size Reduction";}
const std::string Base::impl_name() {return "Base Implementation";}
const std::string Base::param_headers() {return "n:2 m:1 bits:1.58 P:-1";}
std::string Base::get_param_values() {
    std::stringstream ss;
    ss << this->n << " " << this->m << " " <<
          this->prec;
    return ss.str(); 
}

Base::Base() :
    m(0),
    n(0),
    prec(0)
{}

Base::Base(const FusedQRSizeReductionParams& params,
           const ComputationContext& cc) {
    Base::configure(params, cc);
}

void Base::configure(const FusedQRSizeReductionParams& params,
                     const ComputationContext& cc) {
    Matrix B = params.B();
    Matrix R = params.R();
    Matrix U = params.U();
    Matrix tau = params.tau();
    
    assert(B.nrows() != 0 && B.ncols() != 0);
    assert(B.type() == ElementType::MPZ);

    assert(R.ncols() == B.ncols());
    assert(R.nrows() == B.nrows());
    
    assert(U.nrows() == B.ncols());
    assert(U.nrows() == U.ncols());
    assert(U.type() == ElementType::MPZ);

    if (tau.nrows() != 0) {
        assert(tau.nrows() == B.ncols());
        assert(tau.ncols() == 1);
        assert(tau.type() == ElementType::MPFR);
    }

    this->params = params;
    this->B = B;
    this->R = R;
    this->U = U;
    this->tau = tau;

    this->m = B.nrows();
    this->n = B.ncols();
    this->prec = R.prec();
    this->cc = cc;
}

}
}