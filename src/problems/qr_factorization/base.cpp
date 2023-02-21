#include "problems/qr_factorization/base.h"

#include <cassert>
#include <sstream>

namespace flatter {
namespace QRFactorizationImpl {

const std::string Base::prob_name() {return "QR Factorization";}
const std::string Base::impl_name() {return "Base Implementation";}
const std::string Base::param_headers() {return "n:2 m:1 bits:1.58 P:-1";}
std::string Base::get_param_values() {
    std::stringstream ss;
    ss << this->n << " " << this->m << " " <<
          this->prec;
    return ss.str(); 
}

Base::Base() {
}

Base::Base(const Matrix& A, const Matrix& tau, const Matrix& T,
           const ComputationContext& cc) {
    Base::configure(A, tau, T, cc);
}

void Base::configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                    const ComputationContext& cc) {
    assert(T.nrows() == T.ncols());
    
    this->A = A;
    this->tau = tau;
    this->T = T;
    this->m = A.nrows();
    this->n = A.ncols();
    this->rank = std::min(this->m, this->n);
    this->prec = A.prec();
    this->cc = cc;

    assert(T.nrows() == 0 || T.nrows() == this->rank);
}

}
}