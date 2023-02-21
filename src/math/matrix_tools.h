#pragma once

#include "data/matrix.h"
#include "workspace_buffer.h"

namespace flatter {

bool is_approx(mpfr_t& v1, mpfr_t& v2, WorkspaceBuffer<mpfr_t>& wsb);
bool is_matrix_equal(MatrixData<mpfr_t>& M1, MatrixData<mpfr_t>& M2, WorkspaceBuffer<mpfr_t>& wsb);
bool is_matrix_equal(MatrixData<mpz_t>& M1, MatrixData<mpz_t>& M2, WorkspaceBuffer<mpz_t>& wsb);
bool is_matrix_equal(Matrix& M1, Matrix& M2);
bool is_same_gram(MatrixData<mpfr_t>& A, MatrixData<mpfr_t>& B, WorkspaceBuffer<mpfr_t>& ws);
bool is_triangular(MatrixData<mpfr_t>& A);
bool is_same_lattice(MatrixData<mpfr_t>& L1, MatrixData<mpfr_t>& L2);
bool is_same_lattice(const MatrixData<mpz_t>& L1, const MatrixData<mpz_t>& L2);

}