#include "problems/matrix_multiplication.h"

#include <cassert>

#include "aliased.h"
#include "elementary_mpfr.h"
#include "elementary_mpz.h"
#include "elementary_native.h"
#include "elementary_RRZ.h"
#include "elementary_RRl.h"
#include "elementary_ZZl.h"
#include "strassen.h"
#include "threaded.h"

namespace flatter {

using namespace MatrixMultiplicationImpl;

MatrixMultiplication::MatrixMultiplication() {
    _is_configured = false;
}

MatrixMultiplication::MatrixMultiplication(const Matrix& C, const Matrix& A, const Matrix& B,
    const ComputationContext& cc) :
    MatrixMultiplication(C, A, B, false, cc)
{}

MatrixMultiplication::MatrixMultiplication(const Matrix& C, const Matrix& A, const Matrix& B,
    bool accumulate_c, const ComputationContext& cc) :
    MatrixMultiplication(C, A, B, accumulate_c, 10, cc)
{}

MatrixMultiplication::MatrixMultiplication(const Matrix& C, const Matrix& A, const Matrix& B,
    bool accumulate_c, unsigned int cutoff, const ComputationContext& cc) :
    Base(C, A, B, accumulate_c, cc)
{
    this->_is_configured = false;
    this->cutoff = cutoff;
    configure(C, A, B, accumulate_c, cc);
}

MatrixMultiplication::~MatrixMultiplication() {
    if (_is_configured) {
        unconfigure();
    }
}

void MatrixMultiplication::unconfigure() {
    assert(_is_configured);
    delete this->mm;
    _is_configured = false;
}

void MatrixMultiplication::configure(const Matrix& C, const Matrix& A, const Matrix& B,
                                         bool accumulate_c,
                                         const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    Base::configure(C, A, B, accumulate_c, cc);
    mm = nullptr;

    // Check to see if C and A alias or C and B alias
    if (Matrix::is_aliased(C, A) || Matrix::is_aliased(C, B)) {
        mm = new MatrixMultiplicationImpl::Aliased(C, A, B, accumulate_c, cc);
    } else {
        if (cc.is_threaded()) {
            mm = new MatrixMultiplicationImpl::Threaded(C, A, B, accumulate_c, cc);
        } else {
            if (false && m == n && n == k && n > 30) {
                mm = new MatrixMultiplicationImpl::Strassen(C, A, B, accumulate_c, cc);
            } else {
                if (C.type() == ElementType::MPZ &&
                    A.type() == ElementType::MPZ &&
                    B.type() == ElementType::MPZ) {
                    mm = new MatrixMultiplicationImpl::ElementaryMPZ(C, A, B, accumulate_c, cc);
                } else if (C.type() == ElementType::MPFR &&
                    A.type() == ElementType::MPFR &&
                    B.type() == ElementType::MPFR) {
                    mm = new MatrixMultiplicationImpl::ElementaryMPFR(C, A, B, accumulate_c, cc);
                } else if (C.type() == ElementType::MPFR &&
                    A.type() == ElementType::MPFR &&
                    B.type() == ElementType::MPZ) {
                    mm = new MatrixMultiplicationImpl::ElementaryRRZ(C, A, B, accumulate_c, cc);
                } else if (C.type() == ElementType::MPFR &&
                    A.type() == ElementType::MPFR &&
                    B.type() == ElementType::INT64) {
                    mm = new MatrixMultiplicationImpl::ElementaryRRl(C, A, B, accumulate_c, cc);
                } else if (C.type() == ElementType::MPZ &&
                    A.type() == ElementType::MPZ &&
                    B.type() == ElementType::INT64) {
                    mm = new MatrixMultiplicationImpl::ElementaryZZl(C, A, B, accumulate_c, cc);
                } else if (C.type() == ElementType::INT64 &&
                    A.type() == ElementType::INT64 &&
                    B.type() == ElementType::INT64) {
                    mm = new Elementary<int64_t, int64_t, int64_t>(C, A, B, accumulate_c, cc);
                } else if (C.type() == ElementType::DOUBLE &&
                    A.type() == ElementType::DOUBLE &&
                    B.type() == ElementType::DOUBLE) {
                    mm = new Elementary<double, double, double>(C, A, B, accumulate_c, cc);
                } else if (C.type() == ElementType::DOUBLE &&
                    A.type() == ElementType::DOUBLE &&
                    B.type() == ElementType::INT64) {
                    mm = new Elementary<double, double, int64_t>(C, A, B, accumulate_c, cc);
                } else {
                    assert(0);
                }
            }
        }
    }
    assert(mm != nullptr);

    _is_configured = true;
}

void MatrixMultiplication::solve() {
    assert(_is_configured);
    bool use_tasks = cc.is_threaded();

    if (use_tasks && omp_get_active_level() == 0) {
        #pragma omp parallel num_threads(cc.nthreads()) if (use_tasks)
        {
            #pragma omp single
            {
                mm->solve();
            }
        }
    } else {
        #pragma omp taskgroup
        {
            mm->solve();
        }
    }
}

}