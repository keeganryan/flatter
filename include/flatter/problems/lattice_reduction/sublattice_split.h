#pragma once

#include <vector>

namespace flatter {

struct sublattice {
    unsigned int start;
    unsigned int end;
};

class SublatticeSplit {
public:
    SublatticeSplit();
    SublatticeSplit(unsigned int n);
    virtual ~SublatticeSplit();

    virtual std::vector<sublattice> get_sublattices() = 0;
    virtual SublatticeSplit* get_child_split(unsigned int i) = 0;
    virtual void advance_sublattices() = 0;
    virtual bool stopping_point() = 0;

    unsigned int n;
};

}