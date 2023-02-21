#pragma once

#include <mpfr.h>

#include "data/matrix/matrix_data.h"
#include "workspace_buffer.h"

#include "problems/matrix_multiplication.h"

namespace flatter {
namespace MatrixMultiplicationImpl {

class Threaded : public Base {
public:
    Threaded(const Matrix& C, const Matrix& A, const Matrix& B,
             bool accumulate_c,
             const ComputationContext& cc);
    ~Threaded();

    const std::string impl_name();

    void configure(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc);
    void solve();

private:
    void unconfigure();
    bool _is_configured;

    void start_tasks();

    MatrixMultiplication mm;
};

}
}