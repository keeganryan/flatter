#pragma once

#include <mpfr.h>
#include <memory>

#include "matrix_data.h"

namespace flatter {

enum ElementType {
    MPFR,
    MPZ,
    INT64,
    DOUBLE,
};

class Matrix {
public:
    Matrix();
    Matrix(ElementType t, unsigned int m, unsigned int n);
    Matrix(ElementType t, unsigned int m, unsigned int n, unsigned int prec);
    template <class T>
    Matrix(const MatrixData<T>& md);

    template <class T>
    MatrixData<T> data();

    template <class T>
    const MatrixData<T> data() const;

    ElementType type() const;
    unsigned int nrows() const;
    unsigned int ncols() const;
    unsigned int prec() const;
    bool is_transposed() const;
    bool is_identity() const;
    bool is_upper_triangular() const;

    void set_identity();
    void set_precision(unsigned int prec) const;
    Matrix transpose() const;
    Matrix submatrix(unsigned int t, unsigned int d, unsigned int l, unsigned int r);

    static void copy(const Matrix& D, const Matrix& S);
    static void print(const Matrix& A);
    static void save(const Matrix& A, const std::string& fname);
    static bool is_aliased(const Matrix& A, const Matrix& B);

    template<class T>
    static bool is_type(const Matrix& A);

private:
    ElementType t_;
    MatrixData<mpfr_t> md_mpfr;
    MatrixData<mpz_t> md_mpz;
    MatrixData<int64_t> md_int;
    MatrixData<double> md_double;
    std::shared_ptr<void> wsb_;
};

}