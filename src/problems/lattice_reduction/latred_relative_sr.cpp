#include "latred_relative_sr.h"

#include "problems/relative_size_reduction.h"
#include "problems/qr_factorization.h"
#include "problems/lattice_reduction.h"
#include "problems/matrix_multiplication.h"

#include <cassert>

namespace flatter {
namespace LatticeReductionImpl {

const std::string LatRedRelSR::impl_name() {return "LatRedRelSR";}

LatRedRelSR::LatRedRelSR(const LatticeReductionParams& p, const ComputationContext& cc) :
    Base(p, cc)
{
    _is_configured = false;
    configure(p, cc);
}

LatRedRelSR::~LatRedRelSR() {
    if (_is_configured) {
        unconfigure();
    }
}

void LatRedRelSR::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void LatRedRelSR::configure(const LatticeReductionParams& p, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(p, cc);

    assert(p.B2.nrows() != 0);

    _is_configured = true;
}

void LatRedRelSR::solve() {
    log_start();

    LatticeReductionParams p2 = this->params;
    Matrix B2 = this->params.B2;
    Matrix new_B2, new_U2;

    Matrix B_orig(ElementType::MPZ, m, n);
    Matrix B2_orig(ElementType::MPZ, m, B2.ncols());
    Matrix::copy(B_orig, this->params.B());
    Matrix::copy(B2_orig, B2);

    p2.B2 = new_B2;
    p2.U2 = new_U2;
    LatticeReduction lr(p2, cc);
    lr.solve();

    // Get output profile
    Profile prof = p2.L.profile;
    unsigned int prec = 2*(prof.get_spread() + 30);
    int prec2 = B2.prec() - M.prec();
    if (prec2 < 0) {
        prec2 = 0;
    }
    prec = std::max(prec, (unsigned int)prec2);

    Matrix R(ElementType::MPFR, m, n, prec);
    Matrix R2(ElementType::MPFR, m, B2.ncols(), prec);
    Matrix tau(ElementType::MPFR, n, 1, prec);
    Matrix::copy(R, M);

    QRFactorization qr(R, tau, cc);
    qr.solve();

    RelativeSizeReductionParams rsr_params(M, this->params.B2, this->params.U2);
    rsr_params.RV = R;
    rsr_params.R2 = R2;
    rsr_params.tau = tau;

    RelativeSizeReduction rsr(rsr_params, cc);

    rsr.solve();

    MatrixMultiplication mm(this->params.U2, this->params.U(), this->params.U2, cc);
    mm.solve();
    
    log_end();
}

}
}