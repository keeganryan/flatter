#include "data/matrix/matrix_data.h"

#include <cassert>
#include <gmp.h>

namespace flatter {

template <>
unsigned int MatrixData<mpz_t>::prec() const {
    if (m_ == 0 || n_ == 0) {
        return 0;
    }
    unsigned int max_sz = 1;
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int sz = mpz_size(get(i, j));
            max_sz = std::max(max_sz, sz);
        }
    }
    max_sz *= sizeof(mp_limb_t) * 8;
    return max_sz;
}

template <>
bool MatrixData<mpz_t>::is_identity() const {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int expected = (i == j) ? 1 : 0;
            if (mpz_cmp_ui(get(i, j), expected) != 0) {
                return false;
            }
        }
    }
    return true;
}

template <>
bool MatrixData<mpz_t>::is_upper_triangular() const {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (mpz_cmp_ui(get(i, j), 0) != 0) {
                return false;
            }
        }
    }
    return true;
}

template <>
void MatrixData<mpz_t>::set_identity() {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < ncols(); j++) {
            unsigned int expected = (i == j) ? 1 : 0;
            mpz_set_ui(get(i, j), expected);
        }
    }
}

template <>
void MatrixData<mpz_t>::copy(MatrixData<mpz_t>& dst, const MatrixData<mpz_t>& src) {
    assert(dst.nrows() == src.nrows());
    assert(dst.ncols() == src.ncols());
    //assert(dst.prec() == src.prec());

    for (unsigned int i = 0; i < dst.nrows(); i++) {
        for (unsigned int j = 0; j < dst.ncols(); j++) {
            mpz_set(dst(i,j), src(i,j));
        }
    }
}

template<>
void MatrixData<mpz_t>::fprint(FILE* f, const MatrixData<mpz_t>& A) {
    fprintf(f, "[");
    for (unsigned int i = 0; i < A.nrows(); i++) {
        fprintf(f, "[");
        for (unsigned int j = 0; j < A.ncols(); j++) {
            gmp_fprintf(f, "%Zd ", A(i, j));
        }
        fprintf(f, "]\n");
    }
    fprintf(f, "]\n");
}

}