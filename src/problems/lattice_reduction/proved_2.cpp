#include "proved_2.h"

#include "problems/lattice_reduction.h"

namespace flatter {
namespace LatticeReductionImpl {

const std::string Proved2::impl_name() {return "Proved2";}

Proved2::Proved2(const LatticeReductionParams& p, const ComputationContext& cc) :
    Proved3(p, cc)
{}

bool Proved2::is_reduced() {
    if (iterations == 3) {
        return true;
    }
    return false;
}

void Proved2::setup_sublattice_reductions() {

    // Set the indices of the sublattice to reduce.
    unsigned int start, end;
    if (iterations == 0) {
        // Reduce left
        start = 0;
        end = n / 2;
    } else if (iterations == 1) {
        // Reduce right
        start = n / 2;
        end = n;
    } else {
        // Reduce all
        start = 0;
        end = n;
    }

    sublattice_inds.push_back(
        std::make_pair(start, end)
    );

    num_sublattices = 1;

    Matrix B_sub(ElementType::MPZ, end-start, end-start);
    Lattice L_sub(B_sub);
    L_subs.push_back(L_sub);
    Matrix::copy(B_sub, B.submatrix(start, end, start, end));
    L_sub.profile = profile.subprofile(start, end);

    Matrix U_sub(ElementType::MPZ, end-start, end-start);
    U_subs.push_back(U_sub);

    LatticeReductionParams params(B_sub, U_sub, rhf, true);
    params.goal = this->params.goal.subgoal(start, end);
    params.proved = this->params.proved;
    params.offset = this->offset + start;
    params.profile_offset = &global_profile_offsets[start];

    if (iterations == 2) {
        params.lvalid = n / 2;
        params.rvalid = n / 2;
    }
    sub_params.push_back(params);
}

}
}