#pragma once

#include <mpfr.h>

#include <flatter/data/matrix.h>
#include <flatter/problems/problem.h>

namespace flatter {
namespace QRFactorizationImpl {

class Base : public Problem {
public:
    Base();
    Base(const Matrix& A, const Matrix& tau, const Matrix& T, const ComputationContext& cc);
    virtual ~Base() {}

    virtual void configure(const Matrix& A, const Matrix& tau, const Matrix& T, const ComputationContext& cc);

    const std::string prob_name();
    const std::string impl_name();
    const std::string param_headers();
    std::string get_param_values();

protected:
    Matrix A;
    Matrix tau;
    Matrix T;
    unsigned int m;
    unsigned int n;
    unsigned int rank;
    unsigned int prec;
};

}
}