#include "blocked.h"

#include <cassert>

#include "elementary_ZZ.h"
#include "workspace_buffer.h"
#include "problems/size_reduction.h"
#include "problems/matrix_multiplication.h"

namespace flatter {
namespace SizeReductionImpl {

const std::string Blocked::impl_name() {return "Blocked";}

Blocked::Blocked(const Matrix& R, const Matrix& U,
         const ComputationContext& cc) :
    Base(R, U, cc)
{
    _is_configured = false;
    rnd = mpfr_get_default_rounding_mode();
    configure(R, U, cc);
}

Blocked::~Blocked() {
    if (_is_configured) {
        unconfigure();
    }
}

void Blocked::unconfigure() {
    assert(_is_configured);

    wsb->wfree(W.data<mpz_t>().get_data(), bs*bs);
    wsb->wfree(T.data<mpz_t>().get_data(), n*n);
    delete wsb;

    _is_configured = false;
}

void Blocked::configure(const Matrix& R, const Matrix& U,
         const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    assert(R.type() == ElementType::MPZ);
    assert(U.type() == ElementType::MPZ);
    
    Base::configure(R, U, cc);

    unsigned int a = 4;
    bs = n / a;
    bs = std::max(bs, 3u);

    nb_r = (n + bs - 1) / bs;
    nb_c = (n + bs - 1) / bs;

    wsb = new WorkspaceBuffer<mpz_t>(n*n + bs*bs + 3, prec);
    T = MatrixData<mpz_t>(wsb->walloc(n*n), n, n);
    W = MatrixData<mpz_t>(wsb->walloc(bs*bs), bs, bs);

    _is_configured = true;
}

void Blocked::solve() {
    log_start();

    // Set U to identity
    // Set T to 1
    MatrixData<mpz_t> dU = U.data<mpz_t>();
    MatrixData<mpz_t> dT = T.data<mpz_t>();
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            mpz_set_ui(dU(i,j), (i==j)?1:0);
            mpz_set_ui(dT(i,j), 1);
        }
    }

    for (unsigned int diag = 0; diag < nb_c; diag++) {
        update_diagonal(diag);
    }

    log_end();
}

Matrix Blocked::get_tile(Matrix& M, unsigned int i, unsigned int j) {
    assert(i < nb_r);
    assert(j < nb_c);

    unsigned int t = i * bs;
    unsigned int b = std::min((i+1)*bs, n);
    unsigned int l = j * bs;
    unsigned int r = std::min((j+1)*bs, n);

    return M.submatrix(t, b, l, r);
}

void Blocked::update_diagonal(unsigned int diag) {
    assert(diag < nb_c);

    if (diag == 0) {
        for (unsigned int i = 0; i < nb_c; i++) {
            diag_diag(i);  // Updates R(i,i) and U(i,i)
        }
    } else {
        for (unsigned int ind = 0; ind < nb_c - diag; ind++) {
            unsigned int b_i = ind;
            unsigned int b_j = ind + diag;

            diag_above(b_i, b_j); // Update R(i,j)

            for (unsigned int k = b_i + 1; k < b_j; k++) {
                inner_above(b_i, k, b_j); // Update R(i,j) and U(i,j)
            }

            inner_inner(b_i, b_j); // Update R(i,j), U(i,j), T(i,j)
        }
    }
}

void Blocked::diag_diag(unsigned int b_i) {
    // R(i,i) <-
    // U(i,i) <- 

    // When updating a diagonal, update the diagonal
    Matrix R_sub = get_tile(R, b_i, b_i);
    Matrix U_sub = get_tile(U, b_i, b_i);
    Matrix T_sub = get_tile(T, b_i, b_i);

    // Get transformation for diagonal tile
    SizeReduction sr(R_sub, U_sub, cc);
    sr.solve();

    Matrix::copy(T_sub, U_sub);
}

void Blocked::diag_above(unsigned int b_i, unsigned int b_j) {
    // R(i,j) <- U(j,j)

    Matrix R_sub = get_tile(R, b_i, b_j);
    Matrix Tmp = W.submatrix(0, R_sub.nrows(), 0, R_sub.ncols());
    Matrix U_sub = get_tile(U, b_j, b_j);
    Matrix::copy(Tmp, R_sub);
    MatrixMultiplication mm(R_sub, Tmp, U_sub, cc);
    mm.solve();
}

void Blocked::inner_inner(unsigned int b_i, unsigned int b_j) {
    // R(i,j) <- R(i, i), R(i, j)
    // U(i,j) <- U(i, i), R(i, i), R(i, j)
    // T(i,j) <- R(i, i), R(i, j)

    assert(b_i < b_j);

    //printf("Inner inner %u %u\n", b_i, b_j);

    mpz_t* local_ws = wsb->walloc(3);
    mpz_t& mu = local_ws[0];
    mpz_t& tmp = local_ws[1];

    // Assume we have all tiles in column b_i
    // Sets all tiles in column b_j

    Matrix T_sub = get_tile(T, b_i, b_j);
    Matrix R_inner = get_tile(R, b_i, b_j);
    Matrix R_diag = get_tile(R, b_i, b_i);
    Matrix U_inner = get_tile(U, b_i, b_j);
    Matrix U_diag = get_tile(U, b_i, b_i);

    MatrixData<mpz_t> dTs = T_sub.data<mpz_t>();
    MatrixData<mpz_t> dRi = R_inner.data<mpz_t>();
    MatrixData<mpz_t> dRd = R_diag.data<mpz_t>();
    MatrixData<mpz_t> dUi = U_inner.data<mpz_t>();
    MatrixData<mpz_t> dUd = U_diag.data<mpz_t>();

    for (unsigned int j = 0; j < R_inner.ncols(); j++) {
        for (unsigned int ind = 0; ind < R_inner.nrows(); ind++) {
            unsigned int i = R_inner.nrows() - ind - 1;
            // How many copies of R_diag[:,i] do we need to add
            // to size reduce R_inner[i,j] ?
            ElementaryZZ::mpz_div_round(mu, dRi(i, j), dRd(i, i), local_ws + 1);
            
            // Update R_inner, U_inner
            for (unsigned int k = 0; k <= i; k++) {
                mpz_mul(tmp, mu, dRd(k, i));
                mpz_sub(dRi(k, j), dRi(k, j), tmp);
                mpz_mul(tmp, mu, dUd(k, i));
                mpz_sub(dUi(k, j), dUi(k, j), tmp);
            }

            // Update T_sub
            mpz_neg(dTs(i,j), mu);
        }
    }

    wsb->wfree(local_ws, 3);
}

void Blocked::inner_above(unsigned int b_i, unsigned int b_j_l, unsigned int b_j_r) {
    //printf("Inner above %u %u %u\n", b_i, b_j_l, b_j_r);

    Matrix T_sub = get_tile(T, b_j_l, b_j_r);

    Matrix R_left = get_tile(R, b_i, b_j_l);
    Matrix R_right = get_tile(R, b_i, b_j_r);
    Matrix U_left = get_tile(U, b_i, b_j_l);
    Matrix U_right = get_tile(U, b_i, b_j_r);
    
    MatrixMultiplication mm_r(R_right, R_left, T_sub, true, cc);
    mm_r.solve();
    
    MatrixMultiplication mm_u(U_right, U_left, T_sub, true, cc);
    mm_u.solve();

}

}
}