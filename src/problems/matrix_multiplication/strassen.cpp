#include "strassen.h"

#include <cassert>

#include "math/mpfr_blas.h"


namespace flatter {
namespace MatrixMultiplicationImpl {

const std::string Strassen::impl_name() {return "Strassen";}

Strassen::Strassen(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc) :
    Base(C, A, B, accumulate_c, cc)
{
    _is_configured = false;
    configure(C, A, B, accumulate_c, cc);
}

Strassen::~Strassen() {
    unconfigure();
}

void Strassen::unconfigure() {
    assert(_is_configured);

    unsigned int t = (n + 1) / 2;
    wsb->wfree(R.data<mpz_t>().get_data(), t*t);
    wsb->wfree(L.data<mpz_t>().get_data(), t*t);
    wsb->wfree(T.data<mpz_t>().get_data(), t*t);

    delete wsb;
    _is_configured = false;
}

void Strassen::configure(const Matrix& C, const Matrix& A, const Matrix& B,
                         bool accumulate_c,
                         const ComputationContext& cc) {
    assert(C.type() == ElementType::MPZ);
    assert(A.type() == ElementType::MPZ);
    assert(B.type() == ElementType::MPZ);
    assert(!cc.is_threaded());
    
    if (_is_configured) {
        //unconfigure();
        Base::configure(C, A, B, accumulate_c, cc);
        wsb->set_precision(prec);
        return;
    }

    Base::configure(C, A, B, accumulate_c, cc);

    assert(m == n && n == k);
    assert(n >= 2);
    assert(
        (!A.is_transposed() && !B.is_transposed())
    );

    unsigned int t = (n + 1) / 2;
    unsigned int sz_needed = 3 * t * t;
    wsb = new WorkspaceBuffer<mpz_t>(sz_needed, prec);

    dA = this->A.data<mpz_t>();
    dB = this->B.data<mpz_t>();
    dC = this->C.data<mpz_t>();

    T = Matrix(MatrixData<mpz_t>(wsb->walloc(t*t), t, t));
    L = Matrix(MatrixData<mpz_t>(wsb->walloc(t*t), t, t));
    R = Matrix(MatrixData<mpz_t>(wsb->walloc(t*t), t, t));
    mm.configure(T, L, R, false, cc);

    _is_configured = true;
}

void Strassen::solve() {
    assert(_is_configured);
    log_start();

    // Multiply C by beta
    if (!_accumulate_C) {
        for (unsigned int i = 0; i < m; i++) {
            for (unsigned int j = 0; j < n; j++) {
                mpz_set_ui(dC(i, j), 0);
            }
        }
    }

    unsigned int t = (n + 1) / 2;
    
    Matrix A11 = A.submatrix(0, t, 0, t);
    Matrix A12 = A.submatrix(0, t, t, n);
    Matrix A21 = A.submatrix(t, n, 0, t);
    Matrix A22 = A.submatrix(t, n, t, n);

    Matrix B11 = B.submatrix(0, t, 0, t);
    Matrix B12 = B.submatrix(0, t, t, n);
    Matrix B21 = B.submatrix(t, n, 0, t);
    Matrix B22 = B.submatrix(t, n, t, n);

    Matrix C11 = C.submatrix(0, t, 0, t);
    Matrix C12 = C.submatrix(0, t, t, n);
    Matrix C21 = C.submatrix(t, n, 0, t);
    Matrix C22 = C.submatrix(t, n, t, n);

    // M1
    add_padded(L, A11, A22);
    add_padded(R, B11, B22);
    mm.solve();
    add_padded(C11, C11, T);
    add_padded(C22, C22, T);

    // M2
    add_padded(L, A21, A22);
    copy_padded(R, B11);
    mm.solve();
    add_padded(C21, C21, T);
    sub_padded(C22, C22, T);

    // M3
    copy_padded(L, A11);
    sub_padded(R, B12, B22);
    mm.solve();
    add_padded(C12, C12, T);
    add_padded(C22, C22, T);

    // M4
    copy_padded(L, A22);
    sub_padded(R, B21, B11);
    mm.solve();
    add_padded(C11, C11, T);
    add_padded(C21, C21, T);

    // M5
    add_padded(L, A11, A12);
    copy_padded(R, B22);
    mm.solve();
    sub_padded(C11, C11, T);
    add_padded(C12, C12, T);

    // M6
    sub_padded(L, A21, A11);
    add_padded(R, B11, B12);
    mm.solve();
    add_padded(C22, C22, T);

    // M7
    sub_padded(L, A12, A22);
    add_padded(R, B21, B22);
    mm.solve();
    add_padded(C11, C11, T);

    log_end();
}

void Strassen::add_padded(Matrix& R, Matrix& A, Matrix& B) {
    MatrixData<mpz_t> dR = R.data<mpz_t>();
    MatrixData<mpz_t> dA = A.data<mpz_t>();
    MatrixData<mpz_t> dB = B.data<mpz_t>();
    
    for (unsigned int i = 0; i < dR.nrows(); i++) {
        for (unsigned int j = 0; j < dR.ncols(); j++) {
            bool in_a = (i < dA.nrows() && j < dA.ncols());
            bool in_b = (i < dB.nrows() && j < dB.ncols());
            if (in_a && in_b) {
                mpz_add(dR(i, j), dA(i, j), dB(i, j));
            } else if (in_a && !in_b) {
                mpz_set(dR(i, j), dA(i, j));
            } else if (!in_a && in_b) {
                mpz_set(dR(i, j), dB(i, j));
            } else {
                mpz_set_ui(dR(i, j), 0);
            }
        }
    }
}

void Strassen::sub_padded(Matrix& R, Matrix& A, Matrix& B) {
    MatrixData<mpz_t> dR = R.data<mpz_t>();
    MatrixData<mpz_t> dA = A.data<mpz_t>();
    MatrixData<mpz_t> dB = B.data<mpz_t>();
    
    for (unsigned int i = 0; i < R.nrows(); i++) {
        for (unsigned int j = 0; j < R.ncols(); j++) {
            bool in_a = (i < A.nrows() && j < A.ncols());
            bool in_b = (i < B.nrows() && j < B.ncols());
            if (in_a && in_b) {
                mpz_sub(dR(i, j), dA(i, j), dB(i, j));
            } else if (in_a && !in_b) {
                mpz_set(dR(i, j), dA(i, j));
            } else if (!in_a && in_b) {
                mpz_neg(dR(i, j), dB(i, j));
            } else {
                mpz_set_ui(dR(i, j), 0);
            }
        }
    }
}

void Strassen::copy_padded(Matrix& R, Matrix& A) {
    MatrixData<mpz_t> dR = R.data<mpz_t>();
    MatrixData<mpz_t> dA = A.data<mpz_t>();
    
    for (unsigned int i = 0; i < R.nrows(); i++) {
        for (unsigned int j = 0; j < R.ncols(); j++) {
            bool in_a = (i < A.nrows() && j < A.ncols());
            if (in_a) {
                mpz_set(dR(i, j), dA(i, j));
            } else {
                mpz_set_ui(dR(i, j), 0);
            }
        }
    }
}

}
}