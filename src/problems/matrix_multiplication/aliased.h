#pragma once

#include <mpfr.h>

#include "data/matrix/matrix_data.h"
#include "workspace_buffer.h"

#include "problems/matrix_multiplication.h"

namespace flatter {
namespace MatrixMultiplicationImpl {

class Aliased : public Base {
public:
    Aliased(const Matrix& C, const Matrix& A, const Matrix& B,
             bool accumulate_c,
             const ComputationContext& cc);
    ~Aliased();

    const std::string impl_name();

    void configure(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc);
    void solve();

private:
    void unconfigure();

    bool _aliased_with_a;
    bool _aliased_with_b;
    bool _is_configured;

    ElementType a_type;
    ElementType b_type;
    void* wsb_a;
    void* wsb_b;

    Matrix newA;
    Matrix newB;

    MatrixMultiplication mm;
};

}
}