#pragma once

#include <mpfr.h>

#include <flatter/data/matrix.h>
#include <flatter/problems/problem.h>

namespace flatter {
namespace SizeReductionImpl {

class Base : public Problem {
public:
    Base();
    Base(const Matrix& R, const Matrix& U,
         const ComputationContext& cc);
    virtual ~Base() {}

    virtual void configure(const Matrix& R, const Matrix& U,
         const ComputationContext& cc);

    const std::string prob_name();
    const std::string impl_name();
    const std::string param_headers();
    std::string get_param_values();

protected:
    Matrix R;
    Matrix U;

    unsigned int m;
    unsigned int n;
    unsigned int prec;
};

}
}