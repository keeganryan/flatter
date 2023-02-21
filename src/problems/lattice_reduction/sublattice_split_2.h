#pragma once

#include "problems/lattice_reduction/sublattice_split.h"

#include "sublattice_split_3.h"

namespace flatter {

class SubSplitPhase2 : public SublatticeSplit {
public:
    SubSplitPhase2(unsigned int n);
    ~SubSplitPhase2();

    std::vector<sublattice> get_sublattices();
    SublatticeSplit* get_child_split(unsigned int i);
    void advance_sublattices();
    bool stopping_point();

protected:
    unsigned int next_smaller(unsigned int n);
    unsigned int k;
    unsigned int iter;

    SubSplitPhase2* left;
    SubSplitPhase2* right;
    SubSplitPhase3* all;
};

}

