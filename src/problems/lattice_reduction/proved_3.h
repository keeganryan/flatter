#pragma once

#include "recursive_generic.h"

namespace flatter {
namespace LatticeReductionImpl {

class Proved3 : public RecursiveGeneric {
public:
    Proved3(const LatticeReductionParams& p, const ComputationContext& cc);

    const std::string impl_name();

protected:
    virtual bool is_reduced();
    virtual void setup_sublattice_reductions();


    virtual void init_solver();
    virtual void fini_iter();
    virtual unsigned int get_precision_from_spread(double spread);

private:
    void print_profile();
    unsigned long int rval;
};

}
}