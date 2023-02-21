#include "orthogonal_double.h"

#include <cassert>

#include "workspace_buffer.h"

#include "math/mpfr_lapack.h"

#include "orthogonal.h"

typedef int lapack_int;
extern "C" {
    int dlarf_(const char *, int *, int *, double *, int *, double *, double *, int *, double *);
}

namespace flatter {
namespace RelativeSizeReductionImpl {

const std::string OrthogonalDouble::impl_name() {return "OrthogonalDouble";}

OrthogonalDouble::OrthogonalDouble(const RelativeSizeReductionParams& params, const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

OrthogonalDouble::~OrthogonalDouble() {
    if (_is_configured) {
        unconfigure();
    }
}

void OrthogonalDouble::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void OrthogonalDouble::configure(const RelativeSizeReductionParams& params, const ComputationContext& cc) {
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
void OrthogonalDouble::size_reduce(Matrix R, Matrix B, Matrix r_col, Matrix u_col, Matrix b_col, Matrix tau) {
    unsigned int Rr = R.nrows();
    unsigned int Rc = R.ncols();

    assert(B.ncols() == Rc);
    assert(B.nrows() == Rr);
    assert(r_col.nrows() == Rr && r_col.ncols() == 1);
    assert(u_col.nrows() == Rc && u_col.ncols() == 1);
    assert(b_col.nrows() == Rr && b_col.ncols() == 1);
    assert(tau.nrows() == Rc && tau.ncols() == 1);
    assert(R.prec() == r_col.prec() && R.prec() == tau.prec());

    unsigned int lwork = 6 + R.nrows();
    unsigned int prec = R.prec();
    WorkspaceBuffer<double> wsb(lwork + 2, prec);
    double* work = wsb.walloc(lwork);
    double* local = wsb.walloc(2);
    double& c = local[0];
    double& tmp = local[1];
    mpz_t c_int, tmp_int;
    mpz_init(c_int);
    mpz_init(tmp_int);

    MatrixData<double> dR = R.data<double>();
    MatrixData<mpz_t> dB = B.data<mpz_t>();
    MatrixData<mpz_t> db_col = b_col.data<mpz_t>();
    MatrixData<double> dr_col = r_col.data<double>();
    MatrixData<mpz_t> du_col = u_col.data<mpz_t>();
    MatrixData<double> dtau = tau.data<double>();
    
    // Zero out u_col
    for (unsigned int i = 0; i < Rc; i++) {
        mpz_set_ui(du_col(i, 0), 0);
    }

    while (true) {
        // Update the floating point R-factor based on the exact B column
        Matrix::copy(r_col, b_col);

        for (unsigned int i = 0; i < Rc; i++) {
            int m = 1;
            int n = Rr - i;
            double *v = &dR(i,i);
            int incv = dR.stride();
            double *tau = &dtau(i, 0);
            double *C = &dr_col(i, 0);
            int ldc = dr_col.stride();
            double tmp_v = v[0];
            v[0] = 1;
            dlarf_("Right", &m, &n, v, &incv, tau, C, &ldc, work);
            v[0] = tmp_v;
        }

        int max_mu_size = -1;

        // Go through entries in r_col and size reduce
        for (unsigned int i = 0; i < Rc; i++) {
            unsigned int row = Rc - i - 1;
            double& entry = dr_col(row, 0);
            double& diag = dR(row, row);

            c = entry / diag;
            
            if (c > -0.51 && c < 0.51) {
                continue;
            } else {
                c = round(c);
                mpz_set_d(c_int, c);
            }

            // We now have a valid value of c_int
            // for this row. update dr_col, b_col,
            // and u_col
            for (unsigned int j = 0; j <= row; j++) {
                tmp = dR(j, row) * c;
                dr_col(j, 0) -= tmp;
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
    }

    mpz_clear(c_int);
    mpz_clear(tmp_int);
    wsb.wfree(local, 2);
    wsb.wfree(work, lwork);
}

void OrthogonalDouble::solve() {
    log_start();

    unsigned int prec = params.RV.prec();
    MatrixData<double> dR2 = params.R2.data<double>();

    #pragma omp taskloop if (cc.nthreads() > 1)
    for (unsigned int i = 0; i < params.B2.ncols(); i++) {
        {

        assert(prec != 0);
        Matrix r_col(ElementType::DOUBLE, params.B2.nrows(), 1, prec);
        MatrixData<double> dr_col = r_col.data<double>();
        Matrix u_col = params.U.submatrix(0, params.U.nrows(), i, i+1);
        Matrix b_col = params.B2.submatrix(0, params.B2.nrows(), i, i+1);
        size_reduce(params.RV, params.B1, r_col, u_col, b_col, params.tau);

        // Update B_next with the values in r_col, appropriately shifted
        for (unsigned int j = 0; j < params.B2.nrows(); j++) {
            dR2(j, i) = dr_col(j, 0);
        }

        }
    }

    log_end();
}

}
}