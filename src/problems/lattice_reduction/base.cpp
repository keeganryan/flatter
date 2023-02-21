#include "problems/lattice_reduction/base.h"

#include <cassert>
#include <sstream>

namespace flatter {
namespace LatticeReductionImpl {

const std::string Base::prob_name() {return "Lattice Reduction";}
const std::string Base::impl_name() {return "Base Implementation";}
const std::string Base::param_headers() {return "d:3 prec:1 alpha:-1 P:-1";}
std::string Base::get_param_values() {
    std::stringstream ss;
    double p = 0;
    MatrixData<mpz_t> dM = M.data<mpz_t>();
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            double prec = mpz_sizeinbase(dM(i,j), 2);
            p = std::max(prec, p);
        }
    }
    double alpha = params.goal.get_max_drop() / 2;
    if (params.proved) {
        alpha -= HERMITE_BEST_SLOPE;
    } else {
        alpha -= BKZ_BEST_SLOPE;
    }
    ss << this->n << " " << p << " " << alpha;
    return ss.str(); 
}

Base::Base() :
    m(0),
    n(0),
    prec(0)
{}

Base::Base(const LatticeReductionParams& p,
           const ComputationContext& cc) {
    Base::configure(p, cc);
}

void Base::configure(const LatticeReductionParams& p,
                     const ComputationContext& cc) {
    Matrix M = p.B();
    Matrix U = p.U();
    
    assert(M.ncols() == U.nrows());
    assert(U.nrows() == U.ncols());

    this->params = p;
    
    this->M = M;
    this->U = U;
    this->rhf = p.rhf();
    this->profile_offset = p.profile_offset;
    this->offset = p.offset;

    this->lvalid = p.lvalid;
    this->rvalid = p.rvalid;

    this->m = M.nrows();
    this->n = M.ncols();
    this->prec = M.prec();
    this->cc = cc;
}

}
}