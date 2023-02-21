#include "problems/size_reduction/base.h"

#include <cassert>
#include <sstream>

namespace flatter {
namespace SizeReductionImpl {

const std::string Base::prob_name() {return "Size Reduction";}
const std::string Base::impl_name() {return "Base Implementation";}
const std::string Base::param_headers() {return "d:3 bits:1.58 P:-1";}
std::string Base::get_param_values() {
    std::stringstream ss;
    ss << this->n << " " <<
          this->prec;
    return ss.str(); 
}

Base::Base() :
    m(0),
    n(0),
    prec(0)
{}

Base::Base(const Matrix& R, const Matrix& U,
           const ComputationContext& cc) {
    Base::configure(R, U, cc);
}

void Base::configure(const Matrix& R, const Matrix& U,
                     const ComputationContext& cc) {
    assert(R.nrows() == R.ncols());
    assert(R.ncols() == U.nrows());
    assert(U.nrows() == U.ncols());
    assert(R.type() == ElementType::MPZ || R.type() == ElementType::INT64);
    assert(U.type() == ElementType::MPZ || U.type() == ElementType::INT64);
    
    this->R = R;
    this->U = U;

    this->m = R.nrows();
    this->n = R.ncols();
    this->prec = R.prec();
    this->cc = cc;
}

}
}