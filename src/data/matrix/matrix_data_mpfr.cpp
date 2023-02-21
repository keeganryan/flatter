#include "data/matrix/matrix_data.h"

#include<cassert>
#include <mpfr.h>

namespace flatter {

template <>
unsigned int MatrixData<mpfr_t>::prec() const {
    if (m_ == 0 || n_ == 0) {
        return 0;
    }
    return mpfr_get_prec(this->data_[0]);
}

template <>
bool MatrixData<mpfr_t>::is_identity() const {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int expected = (i == j) ? 1 : 0;
            if (mpfr_cmp_ui(get(i, j), expected) != 0) {
                return false;
            }
        }
    }
    return true;
}

template <>
bool MatrixData<mpfr_t>::is_upper_triangular() const {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (mpfr_cmp_ui(get(i, j), 0) != 0) {
                return false;
            }
        }
    }
    return true;
}

template <>
void MatrixData<mpfr_t>::set_identity() {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int expected = (i == j) ? 1 : 0;
            mpfr_set_ui(get(i, j), expected, mpfr_get_default_rounding_mode());
        }
    }
}

template <>
void MatrixData<mpfr_t>::copy(MatrixData<mpfr_t>& dst, const MatrixData<mpfr_t>& src) {
    assert(dst.nrows() == src.nrows());
    assert(dst.ncols() == src.ncols());
    assert(dst.prec() == src.prec());

    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();
    for (unsigned int i = 0; i < dst.nrows(); i++) {
        for (unsigned int j = 0; j < dst.ncols(); j++) {
            mpfr_set(dst(i,j), src(i,j), rnd);
        }
    }
}

template <>
void MatrixData<mpfr_t>::fprint(FILE* f, const MatrixData<mpfr_t>& A) {
    fprintf(f, "[");
    for (unsigned int i = 0; i < A.nrows(); i++) {
        fprintf(f, "[");
        for (unsigned int j = 0; j < A.ncols(); j++) {
            mpfr_fprintf(f, "%Rf, ", A(i, j));
        }
        fprintf(f, "],\n");
    }
    fprintf(f, "]\n");
}

}