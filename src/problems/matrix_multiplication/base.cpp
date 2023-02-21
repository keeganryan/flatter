#include "problems/matrix_multiplication/base.h"

#include <cassert>
#include <sstream>

namespace flatter {
namespace MatrixMultiplicationImpl {

const std::string Base::prob_name() {return "Matrix Multiplication";}
const std::string Base::impl_name() {return "Base Implementation";}
const std::string Base::param_headers() {return "m:1 n:1 k:1 bits:1.58 P:-1";}
std::string Base::get_param_values() {
    std::stringstream ss;
    ss << this->m << " " <<
          this->n << " " <<
          this->k << " " << 
          this->prec;
    return ss.str(); 
}

Base::Base() :
    _accumulate_C(false),
    m(0),
    n(0),
    k(0),
    prec(0)
{}

Base::Base(const Matrix& C, const Matrix& A, const Matrix& B,
           bool accumulate_c,
           const ComputationContext& cc) {
    Base::configure(C, A, B, accumulate_c, cc);
}

void Base::configure(const Matrix& C, const Matrix& A, const Matrix& B,
                     bool accumulate_c,
                     const ComputationContext& cc) {
    assert(C.nrows() == A.nrows());
    assert(C.ncols() == B.ncols());
    assert(A.ncols() == B.nrows());
    assert(C.nrows() != 0);
    assert(C.ncols() != 0);
    assert(B.nrows() != 0);

    this->A = A;
    this->B = B;
    this->C = C;
    this->_accumulate_C = accumulate_c;

    this->m = A.nrows();
    this->n = B.ncols();
    this->k = A.ncols();
    this->prec = A.prec();
    this->cc = cc;
}

}
}