#pragma once

#include <string>

namespace flatter {

template <class T>
class MatrixData {
public:
    MatrixData();
    MatrixData(T* data, unsigned int m, unsigned int n);
    MatrixData(T* data, unsigned int m, unsigned int n, unsigned int stride);
    MatrixData(T* data, unsigned int m, unsigned int n, bool trans, unsigned int stride);

    T* get_data();
    const T& get(unsigned int i, unsigned int j) const;
    T& get(unsigned int i, unsigned int j);
    unsigned int nrows() const;
    unsigned int ncols() const;
    unsigned int prec() const;

    unsigned int stride() const {return stride_;}
    bool is_transposed() const {return transposed_;}
    bool is_identity() const;
    bool is_upper_triangular() const;

    void set_identity();
    MatrixData submatrix(unsigned int t, unsigned int b,
                         unsigned int l, unsigned int r) const;
    MatrixData transpose() const;

    T& operator()(unsigned int i, unsigned int j) {return get(i, j);}
    const T& operator()(unsigned int i, unsigned int j) const {return get(i, j);}

    static void copy(MatrixData<T>& dst, const MatrixData<T>& src);
    static void print(const MatrixData<T>& A);
    static void save(const MatrixData<T>& A, const std::string& fname);
    static void fprint(FILE* file, const MatrixData<T> &A);
    static bool is_aliased(const MatrixData<T>& A, const MatrixData<T>& B);

private:
    T* data_;
    unsigned int m_;
    unsigned int n_;
    unsigned int stride_;
    unsigned int prec_;
    bool transposed_;
};

}