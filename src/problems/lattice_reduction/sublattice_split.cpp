#include "problems/lattice_reduction/sublattice_split.h"

#include <cassert>

namespace flatter {

SublatticeSplit::SublatticeSplit() :
    SublatticeSplit(0)
{}

SublatticeSplit::SublatticeSplit(unsigned int n) :
    n(n)
{
}

SublatticeSplit::~SublatticeSplit() {
}

}