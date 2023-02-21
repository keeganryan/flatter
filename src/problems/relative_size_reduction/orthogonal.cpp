#include "orthogonal.h"

#include <cassert>

#include "workspace_buffer.h"
#include "math/mpfr_lapack.h"

namespace flatter {
namespace RelativeSizeReductionImpl {

const std::string Orthogonal::impl_name() {return "Orthogonal";}

Orthogonal::Orthogonal(const RelativeSizeReductionParams& params, const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

Orthogonal::~Orthogonal() {
    if (_is_configured) {
        unconfigure();
    }
}

void Orthogonal::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Orthogonal::configure(const RelativeSizeReductionParams& params, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
   
    assert(params.B1.type() == ElementType::MPZ);
    assert(params.B2.type() == ElementType::MPZ);
    assert(params.U.type() == ElementType::MPZ);

    assert(params.RV.nrows() == params.B1.nrows());
    assert(params.RV.ncols() == params.B1.ncols());
    assert(params.R2.nrows() == params.B2.nrows());
    assert(params.R2.ncols() == params.B2.ncols());
    assert(params.tau.nrows() == params.RV.ncols());
    assert(params.tau.ncols() == 1);
    
    Base::configure(params, cc);

    _is_configured = true;
}

void Orthogonal::size_reduce(Matrix R, Matrix B, Matrix r_col, Matrix u_col, Matrix b_col, Matrix tau) {
    unsigned int Rr = R.nrows();
    unsigned int Rc = R.ncols();

    assert(B.ncols() == Rc);
    assert(B.nrows() == Rr);
    assert(r_col.nrows() == Rr && r_col.ncols() == 1);
    assert(u_col.nrows() == Rc && u_col.ncols() == 1);
    assert(b_col.nrows() == Rr && b_col.ncols() == 1);
    assert(tau.nrows() == Rc && tau.ncols() == 1);
    assert(R.prec() == r_col.prec() && R.prec() == tau.prec());

    unsigned int lwork = 6;
    unsigned int prec = R.prec();
    WorkspaceBuffer<mpfr_t> wsb(lwork + 2, prec);
    mpfr_t* work = wsb.walloc(lwork);
    mpfr_t* local = wsb.walloc(2);
    mpfr_t& c = local[0];
    mpfr_t& tmp = local[1];
    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
    mpz_t c_int, tmp_int;
    mpz_init(c_int);
    mpz_init(tmp_int);

    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    MatrixData<mpz_t> dB = B.data<mpz_t>();
    MatrixData<mpz_t> db_col = b_col.data<mpz_t>();
    MatrixData<mpfr_t> dr_col = r_col.data<mpfr_t>();
    MatrixData<mpz_t> du_col = u_col.data<mpz_t>();
    MatrixData<mpfr_t> dtau = tau.data<mpfr_t>();
    
    // Zero out u_col
    for (unsigned int i = 0; i < Rc; i++) {
        mpz_set_ui(du_col(i, 0), 0);
    }

    unsigned int iters = 0;
    int prev_max_mu = 1<<30;

    while (true) {
        iters += 1;
        // Update the floating point R-factor based on the exact B column
        Matrix::copy(r_col, b_col);
        // Apply householder vectors to b
        for (unsigned int i = 0; i < Rc; i++) {
            larf(Rr - i, 1, &dR(i,i), dR.stride(), dtau(i, 0), &dr_col(i, 0), dr_col.stride(), work, lwork);
        }

        int max_mu_size = -1;

        // Go through entries in r_col and size reduce
        for (unsigned int i = 0; i < Rc; i++) {
            unsigned int row = Rc - i - 1;
            mpfr_t& entry = dr_col(row, 0);
            mpfr_t& diag = dR(row, row);

            mpfr_div(c, entry, diag, rnd);
            
            if (mpfr_cmp_d(c, -0.51) > 0 && mpfr_cmp_d(c, 0.51) < 0) {
                continue;
            } else {
                int exp = mpfr_get_exp(c);
                max_mu_size = std::max(exp, max_mu_size);
                mpfr_get_z(c_int, c, rnd);
            }

            // We now have a valid value of c_int
            // for this row. update dr_col, b_col,
            // and u_col
            for (unsigned int j = 0; j <= row; j++) {
                mpfr_mul_z(tmp, dR(j, row), c_int, rnd);
                mpfr_sub(dr_col(j, 0), dr_col(j, 0), tmp, rnd);
            }
            for (unsigned int j = 0; j < Rr; j++) {
                mpz_mul(tmp_int, dB(j, row), c_int);
                mpz_sub(db_col(j, 0), db_col(j, 0), tmp_int);
            }
            mpz_sub(du_col(row, 0), du_col(row, 0), c_int);
        }

        if (max_mu_size <= 1) {
            break;
        }
        if (max_mu_size >= prev_max_mu) {
            break;
        }
        prev_max_mu = max_mu_size;
    }

    mpz_clear(c_int);
    mpz_clear(tmp_int);
    wsb.wfree(local, 2);
    wsb.wfree(work, lwork);
}

void Orthogonal::solve() {
    log_start();

    unsigned int prec = params.RV.prec();
    MatrixData<mpfr_t> dR2 = params.R2.data<mpfr_t>();

    #pragma omp taskloop if (cc.nthreads() > 1)
    for (unsigned int i = 0; i < params.B2.ncols(); i++) {
        {

        assert(prec != 0);
        Matrix r_col(ElementType::MPFR, params.B2.nrows(), 1, prec);
        MatrixData<mpfr_t> dr_col = r_col.data<mpfr_t>();
        Matrix u_col = params.U.submatrix(0, params.U.nrows(), i, i+1);
        Matrix b_col = params.B2.submatrix(0, params.B2.nrows(), i, i+1);
        size_reduce(params.RV, params.B1, r_col, u_col, b_col, params.tau);

        // Update B_next with the values in r_col, appropriately shifted
        for (unsigned int j = 0; j < params.B2.nrows(); j++) {
            mpfr_set(dR2(j, i), dr_col(j, 0), mpfr_get_default_rounding_mode());
        }

        }
    }

    log_end();
}

}
}