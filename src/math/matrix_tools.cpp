#include "matrix_tools.h"

#include <cassert>

namespace flatter {

bool is_approx(mpfr_t& v1, mpfr_t& v2, flatter::WorkspaceBuffer<mpfr_t>& wsb) {
    bool is_approx_b = true;
    mpfr_t* local = wsb.walloc(2);
    
    mpfr_t& err = local[0];
    mpfr_t& err_bound = local[1];

    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
    mpfr_prec_t prec = mpfr_get_prec(v1) * 0.8;
    mpfr_set_ui_2exp(err_bound, 1, -prec, rnd);

    mpfr_sub(err, v1, v2, rnd);
    mpfr_abs(err, err, rnd);
    if (!mpfr_zero_p(v2)) {
        mpfr_div(err, err, v2, rnd);
    }

    is_approx_b &= mpfr_number_p(v1);
    is_approx_b &= mpfr_number_p(v2);
    is_approx_b &= (mpfr_cmp(err, err_bound) <= 0);

    wsb.wfree(local, 2);

    return is_approx_b;
}

bool is_matrix_equal(MatrixData<mpfr_t>& M1, MatrixData<mpfr_t>& M2, WorkspaceBuffer<mpfr_t>& wsb) {

    mpfr_t* local_work = wsb.walloc(3);
    mpfr_t& err = local_work[0];
    mpfr_t& max_err = local_work[1];
    mpfr_t& err_bound = local_work[2];

    bool is_matrix_equal = true;

    is_matrix_equal &= (M1.nrows() == M2.nrows());
    is_matrix_equal &= (M1.ncols() == M2.ncols());

    mpfr_set_zero(max_err, 0);
    for (unsigned int i = 0; i < M1.nrows(); i++) {
        for (unsigned int j = 0; j < M1.ncols(); j++) {
            is_matrix_equal &= (mpfr_number_p(M1(i, j)));
            is_matrix_equal &= (mpfr_number_p(M2(i, j)));
            mpfr_sub(err, M1(i, j), M2(i, j), mpfr_get_default_rounding_mode());
            mpfr_abs(err, err, mpfr_get_default_rounding_mode());
            mpfr_max(max_err, max_err, err, mpfr_get_default_rounding_mode());
        }
    }

    mpfr_prec_t prec = mpfr_get_prec(M1(0,0));
    prec = (int)(prec * 0.95);
    mpfr_set_ui_2exp(err_bound, 1, -prec, mpfr_get_default_rounding_mode());

    is_matrix_equal &= (mpfr_cmp(max_err, err_bound) <= 0);


    wsb.wfree(local_work, 3);

    return is_matrix_equal;
}

bool is_matrix_equal(MatrixData<mpz_t>& M1, MatrixData<mpz_t>& M2, WorkspaceBuffer<mpfr_t>& wsb) {
    unsigned int nr = M1.nrows();
    unsigned int nc = M2.nrows();

    if (M2.nrows() != nr || M2.ncols() != nc) {
        return false;
    }

    for (unsigned int i = 0; i < nr; i++) {
        for (unsigned int j = 0; j < nc; j++) {
            if (mpz_cmp(M1(i, j), M2(i, j)) != 0) {
                return false;
            }
        }
    }
    return true;
}

bool is_matrix_equal(Matrix& M1, Matrix& M2) {
    if (M1.type() != M2.type()) {
        return false;
    }

    if (M1.type() == ElementType::MPFR) {
        MatrixData<mpfr_t> dM1 = M1.data<mpfr_t>();
        MatrixData<mpfr_t> dM2 = M2.data<mpfr_t>();

        WorkspaceBuffer<mpfr_t> wsb(3, dM1.prec());
        return is_matrix_equal(dM1, dM2, wsb);
    } else if (M1.type() == ElementType::MPZ) {
        MatrixData<mpz_t> dM1 = M1.data<mpz_t>();
        MatrixData<mpz_t> dM2 = M2.data<mpz_t>();
        WorkspaceBuffer<mpfr_t> wsb(3, dM1.prec());
        return is_matrix_equal(dM1, dM2, wsb);
    }
    return false;
}

bool is_same_gram(MatrixData<mpfr_t>& A, MatrixData<mpfr_t>& B, WorkspaceBuffer<mpfr_t>& ws) {
    bool is_same_gram = true;
    // Ensure that A and B have the same Gram matrix
    mpfr_t& product = *ws.walloc(1);
    mpfr_t& sum_a = *ws.walloc(1);
    mpfr_t& sum_b = *ws.walloc(1);
    mpfr_t& err = *ws.walloc(1);
    mpfr_t& err_bound = *ws.walloc(1);

    is_same_gram &= (A.ncols() == B.ncols());
    unsigned int n_a = A.nrows();
    unsigned int n_b = B.nrows();
    unsigned int m = A.ncols();
    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j <= i; j++) {
            mpfr_set_zero(sum_a, 0);
            for (unsigned int k = 0; k < n_a; k++) {
                mpfr_mul(product, A(k, i), A(k, j), rnd);
                mpfr_add(sum_a, sum_a, product, rnd);
            }

            mpfr_set_zero(sum_b, 0);
            for (unsigned int k = 0; k < n_b; k++) {
                mpfr_mul(product, B(k, i), B(k, j), rnd);
                mpfr_add(sum_b, sum_b, product, rnd);
            }
            is_same_gram &= is_approx(sum_a, sum_b, ws);
        }
    }

    ws.wfree(&err_bound, 1);
    ws.wfree(&err, 1);
    ws.wfree(&sum_b, 1);
    ws.wfree(&sum_a, 1);
    ws.wfree(&product, 1);

    return is_same_gram;
}

bool is_triangular(MatrixData<mpfr_t>& A) {
    bool triangular = true;
    if (A.nrows() != A.ncols()) {
        return false;
    }
    unsigned int n = A.nrows();
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < i; j++) {
            triangular &= (mpfr_zero_p(A.get(i, j)));
        }
    }
    return triangular;
}

bool is_same_lattice(const flatter::MatrixData<mpz_t>& L1, const flatter::MatrixData<mpz_t>& L2) {
  // We want to see if L1 and L2 represent bases of the same lattice.
  // This is the case when there exists a unimodular matrix converting
  // from L1 to L2. We find a matrix converting from L2 to L1, check that
  // it is integer, and check that its inverse is integer.
  
    assert(L1.is_upper_triangular() || L2.is_upper_triangular());
    flatter::MatrixData<mpz_t> A, B;
    flatter::Matrix col(flatter::ElementType::MPZ, L1.nrows(), 1);
    flatter::MatrixData<mpz_t> dcol = col.data<mpz_t>();
    if (L1.is_upper_triangular()) {
        A = L1;
        B = L2;
    } else {
        A = L2;
        B = L1;
    }
    unsigned int m = A.nrows();
    unsigned int n = A.ncols();
    assert(m == n);

    bool is_equal = true;

    mpz_t mul, tmp;
    mpz_init(mul);
    mpz_init(tmp);
    // A is UT.

    for (unsigned int col_id = 0; col_id < n; col_id ++) {
        flatter::MatrixData<mpz_t>::copy(dcol, B.submatrix(0,B.nrows(),col_id,col_id+1));

        for (unsigned int i = 0; i < n; i++) {
            mpz_t& colval = dcol(n - i - 1, 0);
            mpz_t& tval = A(n-i-1, n-i-1);

            mpz_cdiv_q(mul, colval, tval);
            // Subtract
            for (unsigned int j = 0; j < m; j++) {
                mpz_mul(tmp, A(j, n-i-1), mul);
                mpz_sub(dcol(j, 0), dcol(j, 0), tmp);
            }
        }
        // Check for zero
        for (unsigned int j = 0; j < m; j++) {
            if (mpz_cmp_ui(dcol(j, 0), 0) != 0) {
                is_equal = false;
            }
        }
    }

    mpz_clear(mul);
    mpz_clear(tmp);

    return is_equal;
}

}