#include "data/matrix/matrix_data.h"

#include <cassert>
#include <mpfr.h>

namespace flatter {

template <class T>
MatrixData<T>::MatrixData() :
    MatrixData<T>(nullptr, 0, 0, 0)
{}

template <class T>
MatrixData<T>::MatrixData(T* data, unsigned int m, unsigned int n) :
    MatrixData<T>(data, m, n, n)
{}

template <class T>
MatrixData<T>::MatrixData(T* data, unsigned int m, unsigned int n, unsigned int stride) :
    MatrixData<T>(data, m, n, false, stride)
{}

template <class T>
MatrixData<T>::MatrixData(T* data, unsigned int m, unsigned int n, bool trans, unsigned int stride) {
    data_ = data;
    m_ = m;
    n_ = n;
    stride_ = stride;
    transposed_ = trans;
}

template <class T>
T* MatrixData<T>::get_data() {
    return data_;
}

template <class T>
const T& MatrixData<T>::get(unsigned int i, unsigned int j) const {
    assert(i < m_);
    assert(j < n_);
    if (transposed_) {
        return this->data_[j * stride_ + i];
    } else {
        return this->data_[i * stride_ + j];
    }
}

template <class T>
T& MatrixData<T>::get(unsigned int i, unsigned int j) {
    assert(i < m_);
    assert(j < n_);
    if (transposed_) {
        return this->data_[j * stride_ + i];
    } else {
        return this->data_[i * stride_ + j];
    }
}

template <class T>
unsigned int MatrixData<T>::nrows() const {
    return m_;
}

template <class T>
unsigned int MatrixData<T>::ncols() const {
    return n_;
}

template <class T>
bool MatrixData<T>::is_upper_triangular() const {
    for (unsigned int i = 0; i < nrows(); i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (get(i, j) != 0) {
                return false;
            }
        }
    }
    return true;
}

template <class T>
MatrixData<T> MatrixData<T>::submatrix(unsigned int t, unsigned int b,
                                 unsigned int l, unsigned int r) const {
    assert(t < m_ && t <= b && b <= m_);
    assert(l < n_ && l <= r && r <= n_);

    T* ptr = const_cast<T*>(&get(t, l));
    return MatrixData<T>(ptr, b-t, r-l, transposed_, stride_);
}

template <class T>
MatrixData<T> MatrixData<T>::transpose() const {
    MatrixData<T> ret(data_, n_, m_, !transposed_, stride_);
    return ret;
}

template <class T>
void MatrixData<T>::print(const MatrixData<T>& A) {
    MatrixData<T>::fprint(stdout, A);
}

template<class T>
void MatrixData<T>::save(const MatrixData<T>& A, const std::string& fname) {
    FILE* f = fopen(fname.c_str(), "w");
    // Take the transpose, because this is how fplll stores it
    MatrixData<T>::fprint(f, A.transpose());
    fclose(f);
}

template<class T>
bool MatrixData<T>::is_aliased(const MatrixData<T>& A, const MatrixData<T>& B) {
    return A.data_ == B.data_; 
}

template class MatrixData<mpfr_t>;
template class MatrixData<mpz_t>;
template class MatrixData<int64_t>;
template class MatrixData<double>;

}