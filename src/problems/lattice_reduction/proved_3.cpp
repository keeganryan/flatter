#include "proved_3.h"

#include <fstream>

namespace flatter {
namespace LatticeReductionImpl {

const std::string Proved3::impl_name() {return "Proved3";}

Proved3::Proved3(const LatticeReductionParams& p, const ComputationContext& cc) :
    RecursiveGeneric(p, cc)
{}

bool Proved3::is_reduced() {
    if (iterations % 3 == 0 && !lattice_changed) {
        //return true;
    }
    if (iterations == 0) {
        return false;
    }

    return (iterations % 3 == 0 && params.goal.check(profile));
}

unsigned int Proved3::get_precision_from_spread(double spread) {
    return RecursiveGeneric::get_precision_from_spread(spread)*2;
}

void Proved3::setup_sublattice_reductions() {
    // Set the indices of the sublattice to reduce.
    unsigned int start, end;
    if (iterations % 3 == 0) {
        // Reduce middle
        start = n / 4;
        end = 3 * n / 4;

        lattice_changed = false;
    } else if (iterations % 3 == 1) {
        // Reduce left
        start = 0;
        end = n / 2;
    } else {
        // Reduce right
        start = n / 2;
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

    Matrix U_sub(ElementType::MPZ, end-start, end-start);
    U_subs.push_back(U_sub);

    L_sub.profile = profile.subprofile(start, end);

    LatticeReductionParams params(L_sub, U_sub, rhf, true);

    params.goal = this->params.goal.subgoal(start, end);
    params.proved = this->params.proved;
    params.offset = this->offset + start;
    params.profile_offset = &global_profile_offsets[start];
    params.lvalid = n / 4;
    params.rvalid = n / 4;

    sub_params.push_back(params);
}

void Proved3::init_solver() {
    RecursiveGeneric::init_solver();

    unsigned long int r30 = RAND_MAX*rand()+rand();
    unsigned long int s30 = RAND_MAX*rand()+rand();
    unsigned long int t4  = rand() & 0xf;

    rval = (r30 << 34) + (s30 << 4) + t4;
    iterations = -1;
    print_profile();
}

void Proved3::fini_iter() {
    RecursiveGeneric::fini_iter();
    print_profile();
}

void Proved3::print_profile() {
}

}
}