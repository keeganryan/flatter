#pragma once

#include <flatter/data/matrix.h>
#include "params.h"

#include <flatter/problems/problem.h>

namespace flatter {
namespace FusedQRSizeRedImpl {

class Base : public Problem {
public:
    Base();
    Base(const FusedQRSizeReductionParams& params,
         const ComputationContext& cc);
    virtual ~Base() {}

    virtual void configure(const FusedQRSizeReductionParams& params,
        const ComputationContext& cc);

    const std::string prob_name();
    const std::string impl_name();
    const std::string param_headers();
    std::string get_param_values();

protected:
    FusedQRSizeReductionParams params;
    Matrix B;
    Matrix R;
    Matrix U;
    Matrix tau;

    unsigned int m;
    unsigned int n;
    unsigned int prec;
};

}
}