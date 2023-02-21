#pragma once

#include <mpfr.h>

#include "params.h"
#include <flatter/data/matrix.h>
#include <flatter/problems/problem.h>

namespace flatter {
namespace RelativeSizeReductionImpl {

class Base : public Problem {
public:
    Base();
    Base(const RelativeSizeReductionParams& params, 
         const ComputationContext& cc);
    virtual ~Base() {}

    virtual void configure(const RelativeSizeReductionParams& params,
         const ComputationContext& cc);

    const std::string prob_name();
    const std::string impl_name();
    const std::string param_headers();
    std::string get_param_values();

protected:
    RelativeSizeReductionParams params;

    unsigned int n;
};

}
}