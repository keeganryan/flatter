#pragma once

#include "params.h"
#include <flatter/data/matrix.h>
#include <flatter/problems/problem.h>

namespace flatter {
namespace LatticeReductionImpl {

class Base : public Problem {
public:
    Base();
    Base(const LatticeReductionParams& p,
         const ComputationContext& cc);
    virtual ~Base() {}

    virtual void configure(const LatticeReductionParams& p,
         const ComputationContext& cc);

    const std::string prob_name();
    const std::string impl_name();
    const std::string param_headers();
    std::string get_param_values();

protected:
    LatticeReductionParams params;

    Matrix M;
    Matrix U;

    double rhf;

    double* profile_offset;
    unsigned int offset;

    unsigned int lvalid;
    unsigned int rvalid;

    unsigned int m;
    unsigned int n;
    unsigned int prec;
};

}
}