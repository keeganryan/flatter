#include "householder_mpfr.h"

#include <cassert>
#include <omp.h>

#include "math/mpfr_blas.h"
#include "math/mpfr_lapack.h"
#include "workspace_buffer.h"

namespace flatter {
namespace QRFactorizationImpl {

const std::string HouseholderMPFR::impl_name() {return "HouseholderMPFR";}

HouseholderMPFR::HouseholderMPFR(const Matrix& A, const Matrix& tau, const Matrix& T,
                         const ComputationContext& cc) :
    Base(A, tau, T, cc)
{
    _is_configured = false;
    configure(A, tau, T, cc);
}

HouseholderMPFR::~HouseholderMPFR() {
    if (_is_configured) {
        unconfigure();
    }
}

void HouseholderMPFR::unconfigure() {
    if (!_save_tau) {
        wsb->wfree(tau_ptr, rank);
    }
    wsb->wfree(p_ZERO, 1);
    wsb->wfree(work, lwork * cc.nthreads());
    delete wsb;

    _is_configured = false;
}

void HouseholderMPFR::configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                            const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    assert(!A.is_transposed());
    assert(A.type() == ElementType::MPFR);
    assert(tau.type() == ElementType::MPFR);
    assert(T.type() == ElementType::MPFR);

    Base::configure(A, tau, T, cc);

    dA = A.data<mpfr_t>();
    dT = T.data<mpfr_t>();

    lwork = 6; // For larf, larfg
    alloc_size = lwork * cc.nthreads();
    alloc_size += 1;
    if (this->tau.nrows() == 0) {
        _save_tau = false;
        alloc_size += rank;
    } else {
        _save_tau = true;
        MatrixData<mpfr_t> dtau = tau.data<mpfr_t>();
        tau_ptr = dtau.get_data();
    }

    if (T.nrows() == 0) {
        _save_block_reflector = false;
    } else {
        _save_block_reflector = true;
    }
    
    wsb = new WorkspaceBuffer<mpfr_t>(alloc_size, prec);
    work = wsb->walloc(lwork * cc.nthreads());
    p_ZERO = wsb->walloc(1);
    if (!_save_tau) {
        this->tau_ptr = wsb->walloc(rank);
    }

    _is_configured = true;
}

void HouseholderMPFR::solve() {
    log_start();

    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
    mpfr_set_ui(*p_ZERO, 0, rnd);

    for (unsigned int i = 0; i < rank; i++) {
        generate_householder(i);
        apply_householder(i);
    }

    if (_save_block_reflector) {
        mpfr_t* A = this->dA.get_data();
        unsigned int lda = this->dA.stride();
        mpfr_t& aii = work[0];
        mpfr_t& alpha = work[1];

        mpfr_set(dT(0,0), tau_ptr[0], rnd);
        for (unsigned int i = 1; i < rank; i++) {
            mpfr_set(aii, A[i*lda + i], rnd);
            mpfr_set_ui(A[i*lda + i], 1, rnd);

            mpfr_neg(alpha, tau_ptr[i], rnd);
            gemv('T', m-i, i, alpha, &A[i*lda], lda, &A[i*lda + i], lda, *p_ZERO, &dT(0,i), dT.stride(), work+2, lwork-2);
            mpfr_set(A[i*lda + i], aii, rnd);

            trmv('U', 'N', 'N', i, dT.get_data(), dT.stride(), &dT(0, i), dT.stride(), work+2, lwork-2);

            mpfr_set(dT(i, i), tau_ptr[i], rnd);
        }
    }

    if (!_save_tau && !_save_block_reflector) {
        clear_subdiagonal();
    }

    log_end();
}

void HouseholderMPFR::generate_householder(unsigned int i) {
    if (i == A.nrows() - 1) {
        mpfr_set_zero(tau_ptr[i], 0);
        return;
    }
    larfg(m - i, dA(i,i), &dA(i+1,i), this->dA.stride(), tau_ptr[i], work, lwork);
}

void HouseholderMPFR::apply_householder(unsigned int i) {
    if (i == rank - 1) {
        return;
    }
    #pragma omp taskgroup
    {
        for (unsigned int j = 0; j < n - i - 1; j++) {
            #pragma omp task firstprivate(j) shared(i) default(none) if (n > 50 && cc.nthreads() > 1)
            {
                unsigned int tid = 0;
                if (cc.nthreads() != 1) {
                    tid = omp_get_thread_num();
                }
                assert (tid < cc.nthreads());
                mpfr_t* my_work = &work[tid * lwork];
                larf(m-i, 1, &dA(i,i), dA.stride(), tau_ptr[i], &dA(i,i+1+j), dA.stride(), my_work, lwork);
            }
        }
    }
}

void HouseholderMPFR::clear_subdiagonal() {
    // Clear below diagonal
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < i && j < n; j++) {
            mpfr_set_zero(this->dA(i,j), 0);
        }
    }
}

}
}