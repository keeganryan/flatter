#include "data/matrix/matrix_data.h"

#include<cassert>
#include <mpfr.h>

namespace flatter {

template <>
unsigned int MatrixData<double>::prec() const {
    return 53;
}

template <>
bool MatrixData<double>::is_identity() const {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int expected = (i == j) ? 1 : 0;
            if (get(i,j) != expected) {
                return false;
            }
        }
    }
    return true;
}

template <>
bool MatrixData<double>::is_upper_triangular() const {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (get(i, j) != 0) {
                return false;
            }
        }
    }
    return true;
}

template <>
void MatrixData<double>::set_identity() {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int expected = (i == j) ? 1 : 0;
            get(i,j) = expected;
        }
    }
}

template <>
void MatrixData<double>::copy(MatrixData<double>& dst, const MatrixData<double>& src) {
    assert(dst.nrows() == src.nrows());
    assert(dst.ncols() == src.ncols());

    for (unsigned int i = 0; i < dst.nrows(); i++) {
        for (unsigned int j = 0; j < dst.ncols(); j++) {
            dst(i,j) = src(i,j);
        }
    }
}

template <>
void MatrixData<double>::fprint(FILE* f, const MatrixData<double>& A) {
    fprintf(f, "[");
    for (unsigned int i = 0; i < A.nrows(); i++) {
        fprintf(f, "[");
        for (unsigned int j = 0; j < A.ncols(); j++) {
            fprintf(f, "%f, ", A(i, j));
        }
        fprintf(f, "],\n");
    }
    fprintf(f, "]\n");
}

}