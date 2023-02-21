#pragma once

#include <mpfr.h>

#include "data/matrix/matrix_data.h"
#include "workspace_buffer.h"

#include "problems/matrix_multiplication.h"

namespace flatter {
namespace MatrixMultiplicationImpl {

class Strassen : public Base {
public:
    Strassen(const Matrix& C, const Matrix& A, const Matrix& B,
             bool accumulate_c,
             const ComputationContext& cc);
    ~Strassen();

    const std::string impl_name();

    void configure(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc);
    void solve();

private:
    void unconfigure();

    void add_padded(Matrix& R, Matrix& A, Matrix& B);
    void sub_padded(Matrix& R, Matrix& A, Matrix& B);
    void copy_padded(Matrix& R, Matrix& A);

    bool _is_configured;

    WorkspaceBuffer<mpz_t>* wsb;
    MatrixData<mpz_t> dA;
    MatrixData<mpz_t> dB;
    MatrixData<mpz_t> dC;
    Matrix T;
    Matrix R;
    Matrix L;
    MatrixMultiplication mm;
};

}
}