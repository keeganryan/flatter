#pragma once

#include <mpfr.h>

#include <flatter/data/matrix.h>
#include <flatter/problems/problem.h>

namespace flatter {
namespace MatrixMultiplicationImpl {

class Base : public Problem {
public:
    Base();
    Base(const Matrix& C, const Matrix& A, const Matrix& B,
         bool accumulate_c, const ComputationContext& cc);
    virtual ~Base() {}

    virtual void configure(const Matrix& C,
                           const Matrix& A, const Matrix& B,
                           bool accumulate_c,
                           const ComputationContext& cc);

    const std::string prob_name();
    const std::string impl_name();
    const std::string param_headers();
    std::string get_param_values();

protected:
    Matrix C;
    Matrix A;
    Matrix B;
    bool _accumulate_C;

    unsigned int m;
    unsigned int n;
    unsigned int k;
    unsigned int prec;
};

}
}