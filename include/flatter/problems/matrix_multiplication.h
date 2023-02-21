#pragma once

#include "matrix_multiplication/base.h"

namespace flatter {

class MatrixMultiplication : public MatrixMultiplicationImpl::Base {
public:
    MatrixMultiplication();
    MatrixMultiplication(const Matrix& C, const Matrix& A, const Matrix& B,
        const ComputationContext& cc);
    MatrixMultiplication(const Matrix& C, const Matrix& A, const Matrix& B,
        bool accumulate_c,
        const ComputationContext& cc);
    MatrixMultiplication(const Matrix& C, const Matrix& A, const Matrix& B,
        bool accumulate_c,
        unsigned int cutoff,
        const ComputationContext& cc);

    ~MatrixMultiplication();

    void configure(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool _is_configured;
    unsigned int cutoff;
    MatrixMultiplicationImpl::Base* mm;
};

}