#include "data/matrix/matrix_data.h"

#include <cassert>
#include <cinttypes>
#include <gmp.h>

namespace flatter {

template <>
unsigned int MatrixData<int64_t>::prec() const {
    return 63;
}

template <>
bool MatrixData<int64_t>::is_identity() const {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int expected = (i == j) ? 1 : 0;
            if (get(i, j) != expected) {
                return false;
            }
        }
    }
    return true;
}

template <>
void MatrixData<int64_t>::set_identity() {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int expected = (i == j) ? 1 : 0;
            get(i, j) = expected;
        }
    }
}

template <>
void MatrixData<int64_t>::copy(MatrixData<int64_t>& dst, const MatrixData<int64_t>& src) {
    assert(dst.nrows() == src.nrows());
    assert(dst.ncols() == src.ncols());

    for (unsigned int i = 0; i < dst.nrows(); i++) {
        for (unsigned int j = 0; j < dst.ncols(); j++) {
            dst(i,j) = src(i,j);
        }
    }
}

template<>
void MatrixData<int64_t>::fprint(FILE* f, const MatrixData<int64_t>& A) {
    fprintf(f, "[");
    for (unsigned int i = 0; i < A.nrows(); i++) {
        fprintf(f, "[");
        for (unsigned int j = 0; j < A.ncols(); j++) {
            fprintf(f, "%" PRId64 " ", A(i, j));
        }
        fprintf(f, "]\n");
    }
    fprintf(f, "]\n");
}

}