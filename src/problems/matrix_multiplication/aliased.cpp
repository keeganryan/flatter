#include "aliased.h"

#include <cassert>

#include "math/mpfr_blas.h"


namespace flatter {
namespace MatrixMultiplicationImpl {

const std::string Aliased::impl_name() {return "Aliased";}

Aliased::Aliased(const Matrix& C, const Matrix& A, const Matrix& B,
                   bool accumulate_c,
                   const ComputationContext& cc) :
    Base(C, A, B, accumulate_c, cc)
{
    _is_configured = false;
    configure(C, A, B, accumulate_c, cc);
}

Aliased::~Aliased() {
    unconfigure();
}

void Aliased::unconfigure() {
    assert(_is_configured);

    if (_aliased_with_a) {
        unsigned int m = A.nrows();
        unsigned int n = A.ncols();
        if (a_type == ElementType::MPFR) {
            WorkspaceBuffer<mpfr_t>* wsb = (WorkspaceBuffer<mpfr_t>*)this->wsb_a;
            MatrixData<mpfr_t> dM = newA.data<mpfr_t>();
            wsb->wfree(dM.get_data(), m*n);
            delete wsb;
        } else if (a_type == ElementType::MPZ) {
            WorkspaceBuffer<mpz_t>* wsb = (WorkspaceBuffer<mpz_t>*)this->wsb_a;
            MatrixData<mpz_t> dM = newA.data<mpz_t>();
            wsb->wfree(dM.get_data(), m*n);
            delete wsb;
        } else if (a_type == ElementType::INT64) {
            WorkspaceBuffer<int64_t>* wsb = (WorkspaceBuffer<int64_t>*)this->wsb_a;
            MatrixData<int64_t> dM = newA.data<int64_t>();
            wsb->wfree(dM.get_data(), m*n);
            delete wsb;
        } else if (a_type == ElementType::DOUBLE) {
            WorkspaceBuffer<double>* wsb = (WorkspaceBuffer<double>*)this->wsb_a;
            MatrixData<double> dM = newA.data<double>();
            wsb->wfree(dM.get_data(), m*n);
            delete wsb;
        } else {
            assert(0);
        }
    }
    this->wsb_a = nullptr;
    newA = Matrix();

    if (_aliased_with_b) {
        unsigned int m = B.nrows();
        unsigned int n = B.ncols();
        if (b_type == ElementType::MPFR) {
            WorkspaceBuffer<mpfr_t>* wsb = (WorkspaceBuffer<mpfr_t>*)this->wsb_b;
            MatrixData<mpfr_t> dM = newB.data<mpfr_t>();
            wsb->wfree(dM.get_data(), m*n);
            delete wsb;
        } else if (b_type == ElementType::MPZ) {
            WorkspaceBuffer<mpz_t>* wsb = (WorkspaceBuffer<mpz_t>*)this->wsb_b;
            MatrixData<mpz_t> dM = newB.data<mpz_t>();
            wsb->wfree(dM.get_data(), m*n);
            delete wsb;
        } else if (b_type == ElementType::INT64) {
            WorkspaceBuffer<int64_t>* wsb = (WorkspaceBuffer<int64_t>*)this->wsb_b;
            MatrixData<int64_t> dM = newB.data<int64_t>();
            wsb->wfree(dM.get_data(), m*n);
            delete wsb;
        } else {
            assert(0);
        }
    }
    this->wsb_b = nullptr;
    newB = Matrix();

    _is_configured = false;
}

void Aliased::configure(const Matrix& C, const Matrix& A, const Matrix& B,
                        bool accumulate_c,
                        const ComputationContext& cc) {
    _aliased_with_a = Matrix::is_aliased(C, A);
    _aliased_with_b = Matrix::is_aliased(C, B);
    assert(_aliased_with_a || _aliased_with_b);
    
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(C, A, B, accumulate_c, cc);

    if (_aliased_with_a) {
        this->a_type = A.type();
        unsigned int m = A.nrows();
        unsigned int n = A.ncols();

        if (a_type == ElementType::MPFR) {
            WorkspaceBuffer<mpfr_t>* wsb_a = new WorkspaceBuffer<mpfr_t>(m*n, A.prec());
            MatrixData<mpfr_t> dM = MatrixData<mpfr_t>(wsb_a->walloc(m*n), m, n);
            this->wsb_a = wsb_a;
            this->newA = Matrix(dM);
        } else if (a_type == ElementType::MPZ) {
            WorkspaceBuffer<mpz_t>* wsb_a = new WorkspaceBuffer<mpz_t>(m*n, A.prec());
            MatrixData<mpz_t> dM = MatrixData<mpz_t>(wsb_a->walloc(m*n), m, n);
            this->wsb_a = wsb_a;
            this->newA = Matrix(dM);
        } else if (a_type == ElementType::INT64) {
            WorkspaceBuffer<int64_t>* wsb_a = new WorkspaceBuffer<int64_t>(m*n, A.prec());
            MatrixData<int64_t> dM = MatrixData<int64_t>(wsb_a->walloc(m*n), m, n);
            this->wsb_a = wsb_a;
            this->newA = Matrix(dM);
        } else if (a_type == ElementType::DOUBLE) {
            WorkspaceBuffer<double>* wsb_a = new WorkspaceBuffer<double>(m*n, A.prec());
            MatrixData<double> dM = MatrixData<double>(wsb_a->walloc(m*n), m, n);
            this->wsb_a = wsb_a;
            this->newA = Matrix(dM);
        } else {
            assert(0);
        }
    } else {
        newA = A;
    }
    if (_aliased_with_b) {
        this->b_type = B.type();
        unsigned int m = B.nrows();
        unsigned int n = B.ncols();

        if (b_type == ElementType::MPFR) {
            WorkspaceBuffer<mpfr_t>* wsb = new WorkspaceBuffer<mpfr_t>(m*n, B.prec());
            MatrixData<mpfr_t> dM = MatrixData<mpfr_t>(wsb->walloc(m*n), m, n);
            this->wsb_b = wsb;
            this->newB = Matrix(dM);
        } else if (b_type == ElementType::MPZ) {
            WorkspaceBuffer<mpz_t>* wsb = new WorkspaceBuffer<mpz_t>(m*n, B.prec());
            MatrixData<mpz_t> dM = MatrixData<mpz_t>(wsb->walloc(m*n), m, n);
            this->wsb_b = wsb;
            this->newB = Matrix(dM);
        } else if (b_type == ElementType::INT64) {
            WorkspaceBuffer<int64_t>* wsb_b = new WorkspaceBuffer<int64_t>(m*n, A.prec());
            MatrixData<int64_t> dM = MatrixData<int64_t>(wsb_b->walloc(m*n), m, n);
            this->wsb_b = wsb_b;
            this->newB = Matrix(dM);
        } else {
            assert(0);
        }
    } else {
        newB = B;
    }

    mm.configure(C, newA, newB, accumulate_c, cc);

    _is_configured = true;
}

void Aliased::solve() {
    assert(_is_configured);
    //log_start();

    if (_aliased_with_a) {
        Matrix::copy(newA, A);
    }
    if (_aliased_with_b) {
        Matrix::copy(newB, B);
    }
    mm.solve();

    //log_end();
}

}
}