#include "problems/lattice_reduction/params.h"


#include <cstdlib>

namespace flatter {

LatticeReductionParams::LatticeReductionParams() {
    _is_upper_triangular = false;

    profile_offset = nullptr;
    offset = 0;

    lvalid = 0;
    rvalid = 0;

    phase = 1;
    log_cond = 0;
    aggressive_precision = false;
}

LatticeReductionParams::LatticeReductionParams(Lattice& L, Matrix& U, double rhf, bool is_upper_triangular) {
    this->L = L;
    U_ = U;
    rhf_ = rhf;
    goal = LatticeReductionGoal::from_RHF(this->L.rank(), rhf);

    proved = false;
    char* env_proved = std::getenv("FLATTER_PROVED");
    if (env_proved != nullptr) {
        proved = true;
    }

    _is_upper_triangular = is_upper_triangular;

    profile_offset = nullptr;
    offset = 0;

    lvalid = 0;
    rvalid = 0;
    split = nullptr;

    phase = 0;
    log_cond = 0;
    aggressive_precision = false;
}

LatticeReductionParams::LatticeReductionParams(const Matrix& B, const Matrix& U, double rhf, bool is_upper_triangular) {
    this->L = Lattice(B.ncols(), B.nrows());
    L.basis() = B;
    U_ = U;
    rhf_ = rhf;
    goal = LatticeReductionGoal::from_RHF(L.rank(), rhf);

    proved = false;
    char* env_proved = std::getenv("FLATTER_PROVED");
    if (env_proved != nullptr) {
        proved = true;
    }

    _is_upper_triangular = is_upper_triangular;

    profile_offset = nullptr;
    offset = 0;

    lvalid = 0;
    rvalid = 0;
    split = nullptr;

    phase = 0;
    log_cond = 0;
    aggressive_precision = false;
}

}