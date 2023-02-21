#include "data/matrix/matrix.h"

#include "workspace_buffer.h"
#include <cassert>

namespace flatter {

Matrix::Matrix() {
    t_ = ElementType::MPFR;
    md_mpfr = MatrixData<mpfr_t>();
    wsb_ = nullptr;
}

Matrix::Matrix(ElementType t, unsigned int m, unsigned int n) :
    Matrix(t, m, n, 0)
{}

Matrix::Matrix(ElementType t, unsigned int m, unsigned int n, unsigned int prec) {
    t_ = t;
    if (t == ElementType::MPFR) {
        WorkspaceBuffer<mpfr_t>* wsb = new WorkspaceBuffer<mpfr_t>(m*n, prec);
        MatrixData<mpfr_t> md (wsb->walloc(m*n), m, n);
        md_mpfr = md;
        wsb_ = std::shared_ptr<void>(wsb);
    } else if (t == ElementType::MPZ) {
        WorkspaceBuffer<mpz_t>* wsb = new WorkspaceBuffer<mpz_t>(m*n, prec);
        MatrixData<mpz_t> md (wsb->walloc(m*n), m, n);
        md_mpz = md;
        wsb_ = std::shared_ptr<void>(wsb);
    } else if (t == ElementType::INT64) {
        WorkspaceBuffer<int64_t>* wsb = new WorkspaceBuffer<int64_t>(m*n, prec);
        MatrixData<int64_t> md (wsb->walloc(m*n), m, n);
        md_int = md;
        wsb_ = std::shared_ptr<void>(wsb);
    } else if (t == ElementType::DOUBLE) {
        WorkspaceBuffer<double>* wsb = new WorkspaceBuffer<double>(m*n, prec);
        MatrixData<double> md (wsb->walloc(m*n), m, n);
        md_double = md;
        wsb_ = std::shared_ptr<void>(wsb);
    } else {
        assert(0);
    }
}

template<>
Matrix::Matrix(const MatrixData<mpfr_t>& md) {
    t_ = ElementType::MPFR;
    md_mpfr = md;
    wsb_ = nullptr;
}

template<>
Matrix::Matrix(const MatrixData<mpz_t>& md) {
    t_ = ElementType::MPZ;
    md_mpz = md;
    wsb_ = nullptr;
}

template<>
Matrix::Matrix(const MatrixData<int64_t>& md) {
    t_ = ElementType::INT64;
    md_int = md;
    wsb_ = nullptr;
}

template<>
Matrix::Matrix(const MatrixData<double>& md) {
    t_ = ElementType::DOUBLE;
    md_double = md;
    wsb_ = nullptr;
}

template<>
MatrixData<mpfr_t> Matrix::data<mpfr_t>() {
    assert(t_ == ElementType::MPFR);
    return md_mpfr;
}
template<>
MatrixData<mpz_t> Matrix::data<mpz_t>() {
    assert(t_ == ElementType::MPZ);
    return md_mpz;
}
template<>
MatrixData<int64_t> Matrix::data<int64_t>() {
    assert(t_ == ElementType::INT64);
    return md_int;
}
template<>
MatrixData<double> Matrix::data<double>() {
    assert(t_ == ElementType::DOUBLE);
    return md_double;
}

template<>
const MatrixData<mpfr_t> Matrix::data<mpfr_t>() const {
    assert(t_ == ElementType::MPFR);
    return md_mpfr;
}
template<>
const MatrixData<mpz_t> Matrix::data<mpz_t>() const {
    assert(t_ == ElementType::MPZ);
    return md_mpz;
}
template<>
const MatrixData<int64_t> Matrix::data<int64_t>() const {
    assert(t_ == ElementType::INT64);
    return md_int;
}
template<>
const MatrixData<double> Matrix::data<double>() const {
    assert(t_ == ElementType::DOUBLE);
    return md_double;
}

ElementType Matrix::type() const {
    return t_;
}

#define MDCALL(type, fname)                                     \
type Matrix::fname() const {                                    \
    if (t_ == ElementType::MPFR) {                              \
        return md_mpfr.fname();                                 \
    } else if (t_ == ElementType::MPZ) {                        \
        return md_mpz.fname();                                  \
    } else if (t_ == ElementType::INT64) {                      \
        return md_int.fname();                                  \
    } else if (t_ == ElementType::DOUBLE) {                     \
        return md_double.fname();                               \
    } else {                                                    \
        assert(0);                                              \
        return md_mpfr.fname();                                 \
    }                                                           \
}

MDCALL(unsigned int, nrows)
MDCALL(unsigned int, ncols)
MDCALL(unsigned int, prec)
MDCALL(bool, is_transposed)
MDCALL(bool, is_identity)
MDCALL(bool, is_upper_triangular)
                                                            
void Matrix::set_identity() {        
    if (type() == ElementType::MPFR) {
        md_mpfr.set_identity();
    } else if (type() == ElementType::MPZ) {
        md_mpz.set_identity();
    } else if (type() == ElementType::INT64) {
        md_int.set_identity();
    } else if (type() == ElementType::DOUBLE) {
        md_double.set_identity();
    } else {
        assert(0);
    }
}

void Matrix::set_precision(unsigned int prec) const {
    assert(wsb_ != nullptr);

    if (type() == ElementType::MPFR) {
        auto wsb = std::static_pointer_cast<WorkspaceBuffer<mpfr_t>>(wsb_);
        wsb->set_precision(prec);
    }
}

Matrix Matrix::transpose() const {
    switch (t_) {
    case ElementType::MPFR:
        return Matrix(md_mpfr.transpose());
    case ElementType::MPZ:
        return Matrix(md_mpz.transpose());
    case ElementType::INT64:
        return Matrix(md_int.transpose());
    case ElementType::DOUBLE:
        return Matrix(md_double.transpose());    
    }
    return Matrix();
}

Matrix Matrix::submatrix(unsigned int t, unsigned int b,
                         unsigned int l, unsigned int r) {
    switch (t_) {
    case ElementType::MPFR:
        return Matrix(md_mpfr.submatrix(t, b, l, r));
    case ElementType::MPZ:
        return Matrix(md_mpz.submatrix(t, b, l, r));
    case ElementType::INT64:
        return Matrix(md_int.submatrix(t, b, l, r));
    case ElementType::DOUBLE:
        return Matrix(md_double.submatrix(t, b, l, r));
    }
    return Matrix();
}

void Matrix::copy(const Matrix& D, const Matrix& S) {
    assert(D.nrows() == S.nrows());
    assert(D.ncols() == S.ncols());
    if (D.type() == ElementType::MPFR && S.type() == ElementType::MPFR) {
        MatrixData<mpfr_t> dD = D.data<mpfr_t>();
        MatrixData<mpfr_t>::copy(dD, S.data<mpfr_t>());
    } else if (D.type() == ElementType::MPZ && S.type() == ElementType::MPZ) {
        MatrixData<mpz_t> dD = D.data<mpz_t>();
        MatrixData<mpz_t>::copy(dD, S.data<mpz_t>());
    } else if (D.type() == ElementType::INT64 && S.type() == ElementType::INT64) {
        MatrixData<int64_t> dD = D.data<int64_t>();
        MatrixData<int64_t>::copy(dD, S.data<int64_t>());
    } else if (D.type() == ElementType::DOUBLE && S.type() == ElementType::DOUBLE) {
        MatrixData<double> dD = D.data<double>();
        MatrixData<double>::copy(dD, S.data<double>());
    } else if (D.type() == ElementType::MPFR && S.type() == ElementType::MPZ) {
        mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
        MatrixData<mpfr_t> dD = D.data<mpfr_t>();
        MatrixData<mpz_t> dS = S.data<mpz_t>();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                mpfr_set_z(dD(i,j), dS(i,j), rnd);
            }
        }
    } else if (D.type() == ElementType::MPFR && S.type() == ElementType::INT64) {
        mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
        MatrixData<mpfr_t> dD = D.data<mpfr_t>();
        MatrixData<int64_t> dS = S.data<int64_t>();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                mpfr_set_si(dD(i,j), dS(i,j), rnd);
            }
        }
    } else if (D.type() == ElementType::INT64 && S.type() == ElementType::MPZ) {
        MatrixData<int64_t> dD = D.data<int64_t>();
        MatrixData<mpz_t> dS = S.data<mpz_t>();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                dD(i,j) = mpz_get_si(dS(i,j));
            }
        }
    } else if (D.type() == ElementType::MPZ && S.type() == ElementType::INT64) {
        MatrixData<mpz_t> dD = D.data<mpz_t>();
        MatrixData<int64_t> dS = S.data<int64_t>();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                mpz_set_si(dD(i,j), dS(i,j));
            }
        }
    } else if (D.type() == ElementType::DOUBLE && S.type() == ElementType::INT64) {
        MatrixData<double> dD = D.data<double>();
        MatrixData<int64_t> dS = S.data<int64_t>();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                dD(i,j) = dS(i,j);
            }
        }
    } else if (D.type() == ElementType::DOUBLE && S.type() == ElementType::MPZ) {
        MatrixData<double> dD = D.data<double>();
        MatrixData<mpz_t> dS = S.data<mpz_t>();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                dD(i,j) = mpz_get_d(dS(i,j));
            }
        }
    } else if (D.type() == ElementType::MPZ && S.type() == ElementType::MPFR) {
        MatrixData<mpz_t> dD = D.data<mpz_t>();
        MatrixData<mpfr_t> dS = S.data<mpfr_t>();

        mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                mpfr_get_z(dD(i,j), dS(i,j), rnd);
            }
        }
    } else if (D.type() == ElementType::DOUBLE && S.type() == ElementType::MPFR) {
        MatrixData<double> dD = D.data<double>();
        MatrixData<mpfr_t> dS = S.data<mpfr_t>();

        mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                dD(i,j) = mpfr_get_d(dS(i,j), rnd);
            }
        }
    } else if (D.type() == ElementType::MPFR && S.type() == ElementType::DOUBLE) {
        MatrixData<mpfr_t> dD = D.data<mpfr_t>();
        MatrixData<double> dS = S.data<double>();

        mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                mpfr_set_d(dD(i,j), dS(i,j), rnd);
            }
        }
    } else if (D.type() == ElementType::MPZ && S.type() == ElementType::DOUBLE) {
        MatrixData<mpz_t> dD = D.data<mpz_t>();
        MatrixData<double> dS = S.data<double>();

        for (unsigned int i = 0; i < D.nrows(); i++) {
            for (unsigned int j = 0; j < D.ncols(); j++) {
                mpz_set_d(dD(i,j), dS(i,j));
            }
        }
    } else {
        //printf("Bad types %d %d\n", D.type(), S.type());
        assert(false);
    }
}

void Matrix::print(const Matrix& A) {
    if (A.type() == ElementType::MPFR) {
        MatrixData<mpfr_t>::print(A.data<mpfr_t>());
    } else if (A.type() == ElementType::MPZ) {
        MatrixData<mpz_t>::print(A.data<mpz_t>());
    } else if (A.type() == ElementType::INT64) {
        MatrixData<int64_t>::print(A.data<int64_t>());
    } else if (A.type() == ElementType::DOUBLE) {
        MatrixData<double>::print(A.data<double>());
    } else {
        assert(0);
    }
}

void Matrix::save(const Matrix& A, const std::string& fname) {
    if (A.type() == ElementType::MPFR) {
        MatrixData<mpfr_t>::save(A.data<mpfr_t>(), fname);
    } else if (A.type() == ElementType::MPZ) {
        MatrixData<mpz_t>::save(A.data<mpz_t>(), fname);
    } else {
        assert(0);
    }
}

bool Matrix::is_aliased(const Matrix& A, const Matrix& B) {
    // Do A and B overlap on the underlying data?
    if (A.type() != B.type()) {
        return false;
    }
    if (A.type() == ElementType::MPFR) {
        return MatrixData<mpfr_t>::is_aliased(A.data<mpfr_t>(), B.data<mpfr_t>());
    } else if (A.type() == ElementType::MPZ) {
        return MatrixData<mpz_t>::is_aliased(A.data<mpz_t>(), B.data<mpz_t>());
    } else if (A.type() == ElementType::INT64) {
        return MatrixData<int64_t>::is_aliased(A.data<int64_t>(), B.data<int64_t>());
    } else if (A.type() == ElementType::DOUBLE) {
        return MatrixData<double>::is_aliased(A.data<double>(), B.data<double>());
    } else {
        assert(0);
        return false;
    }
}

template <>
bool Matrix::is_type<int64_t>(const Matrix& A) {
    return A.type() == ElementType::INT64;
}
template <>
bool Matrix::is_type<double>(const Matrix& A) {
    return A.type() == ElementType::DOUBLE;
}

}