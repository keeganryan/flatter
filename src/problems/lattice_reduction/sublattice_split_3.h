#pragma once

#include "problems/lattice_reduction/sublattice_split.h"

namespace flatter {

class SubSplitPhase3 : public SublatticeSplit {
public:
    SubSplitPhase3(unsigned int n);
    ~SubSplitPhase3();

    std::vector<sublattice> get_sublattices();
    SublatticeSplit* get_child_split(unsigned int i);
    void advance_sublattices();
    bool stopping_point();

protected:
    friend class SubSplitPhase2;
    
    SubSplitPhase3(unsigned int n, unsigned int k);
    SubSplitPhase3(SubSplitPhase3* l, SubSplitPhase3* r);

    void reset();
    unsigned int cycle_len();

    unsigned int k;
    SubSplitPhase3* left_child;
    SubSplitPhase3* mid_child;
    SubSplitPhase3* right_child;

    unsigned int iter;
};

}

