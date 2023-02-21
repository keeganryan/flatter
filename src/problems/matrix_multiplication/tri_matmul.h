#pragma once

#include <mpfr.h>

#include "data/matrix.h"
#include "workspace_buffer.h"
#include "computation_context.h"

namespace flatter {

class TriMatrixMultiplication {
public:
    TriMatrixMultiplication(MatrixData<mpfr_t>& A, MatrixData<mpfr_t>& B,
                            char side, char uplo, char transa, char diag,
                            const mpfr_t* alpha,
                            WorkspaceBuffer<mpfr_t>* wsb,
                            const ComputationContext& cc);
    ~TriMatrixMultiplication();
                             
    void solve();

protected:
    MatrixData<mpfr_t> A;
    MatrixData<mpfr_t> B;
    char side;
    char uplo;
    char transa;
    char diag;
    const mpfr_t* alpha;
    mpfr_t* work;

    WorkspaceBuffer<mpfr_t>* wsb;
    const ComputationContext* cc;
};

}