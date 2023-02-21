#include "threaded.h"

#include <cassert>

#include <omp.h>

#include "math/mpfr_blas.h"


namespace flatter {
namespace MatrixMultiplicationImpl {

const std::string Threaded::impl_name() {return "Threaded";}

Threaded::Threaded(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc) :
    Base(C, A, B, accumulate_c, cc)
{
    _is_configured = false;
    configure(C, A, B, accumulate_c, cc);
}

Threaded::~Threaded() {
    unconfigure();
}

void Threaded::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Threaded::configure(const Matrix& C, const Matrix& A, const Matrix& B,
                        bool accumulate_c,
                        const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(C, A, B, accumulate_c, cc);
    _is_configured = true;
}

void Threaded::start_tasks() {
    ComputationContext single_cc(1);

    double unit_cost = A.ncols() * prec;
    unsigned int n_est = 20000000 / unit_cost;
    n_est = sqrt(n_est);
    n_est = std::max(n_est, 1u);

    for (unsigned int i = 0; i < A.nrows(); i += n_est) {
        for (unsigned int j = 0; j < B.ncols(); j += n_est) {
            #pragma omp task firstprivate(i, j)
            {
                unsigned int top = i;
                unsigned int bot = std::min(i + n_est, A.nrows());
                unsigned int lef = j;
                unsigned int rig = std::min(j + n_est, B.ncols());

                Matrix subC = C.submatrix(top, bot, lef, rig);
                Matrix subA = A.submatrix(top, bot, 0, A.ncols());
                Matrix subB = B.submatrix(0, B.nrows(), lef, rig);

                MatrixMultiplication mm(subC, subA, subB, _accumulate_C, single_cc);
                mm.solve();
            }
        }
    }
    #pragma omp taskwait
}

void Threaded::solve() {
    assert(_is_configured);
    log_start();
    
    if (omp_get_active_level() == 0) {
        #pragma omp parallel num_threads(cc.nthreads())
        {
            #pragma omp single
            {
                start_tasks();
            }
        }
    } else {
        #pragma omp taskgroup
        start_tasks();
    }

    log_end();
}

}
}