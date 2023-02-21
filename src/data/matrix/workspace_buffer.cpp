#include "workspace_buffer.h"

#include <cassert>
#include <cstdlib>
#include <mpfr.h>

namespace flatter {

template<>
void WorkspaceBuffer<mpfr_t>::ws_delete(mpfr_t* ptr, unsigned int sz) {
    for (unsigned int i = 0; i < sz; i++) {
        mpfr_clear(ptr[i]);
    }
    delete[] ptr;
}

template <>
WorkspaceBuffer<mpfr_t>::WorkspaceBuffer(unsigned int sz, unsigned int prec) {
    this->offset_ = 0;
    this->prec_ = prec;

    mpfr_t* buf = new mpfr_t[sz];
    for (unsigned int i = 0; i < sz; i++) {
        mpfr_init2(buf[i], this->prec_);
    }
    this->ws_ = std::shared_ptr<mpfr_t[]>(
        buf,
        [=](mpfr_t* p) { ws_delete(p, sz); }
    );

    sz_ = sz;
}

template <>
void WorkspaceBuffer<mpfr_t>::set_precision(unsigned int prec) {
    if (prec != this->prec_) {
        for (unsigned int i = 0; i < sz_; i++) {
            mpfr_prec_round(ws_[i], prec, mpfr_get_default_rounding_mode());
        }
        this->prec_ = prec;
    }
}

template<>
void WorkspaceBuffer<mpz_t>::ws_delete(mpz_t* ptr, unsigned int sz) {
    for (unsigned int i = 0; i < sz; i++) {
        mpz_clear(ptr[i]);
    }
    delete[] ptr;
}

template <>
WorkspaceBuffer<mpz_t>::WorkspaceBuffer(unsigned int sz, unsigned int prec) {
    this->offset_ = 0;
    prec = 0;
    this->prec_ = prec;

    mpz_t* buf = new mpz_t[sz];
    for (unsigned int i = 0; i < sz; i++) {
        mpz_init2(buf[i], this->prec_);
    }
    this->ws_ = std::shared_ptr<mpz_t[]>(
        buf,
        [=](mpz_t* p) { ws_delete(p, sz); }
    );
    sz_ = sz;
}

template <>
void WorkspaceBuffer<mpz_t>::set_precision(unsigned int prec) {
    if (prec != this->prec_) {
        for (unsigned int i = 0; i < sz_; i++) {
            mpz_realloc2(ws_[i], prec);
        }
        this->prec_ = prec;
    }
}

template <class T>
WorkspaceBuffer<T>::WorkspaceBuffer(unsigned int sz, unsigned int prec) {
    this->offset_ = 0;
    this->prec_ = prec;

    this->ws_ = std::shared_ptr<T[]>(
        new T[sz],
        [=](T* p) { ws_delete(p, sz); }
    );
    
    sz_ = sz;
}

template <class T>
void WorkspaceBuffer<T>::set_precision(unsigned int prec) {
}

template <class T>
T* WorkspaceBuffer<T>::walloc(unsigned int sz) {
    // Do we have to grow?
    T* ret;

    assert(sz_ - offset_ >= sz);

    ret = &ws_[offset_];
    offset_ += sz;
    return ret;
}

template <class T>
void WorkspaceBuffer<T>::wfree(T* ptr, unsigned int sz) {
    assert(offset_ >= sz);
    // Out-of-order workspace free?
    assert(&ws_[0] + offset_ - sz == ptr);
    offset_ -= sz;
}

template <class T>
void WorkspaceBuffer<T>::ws_delete(T* ptr, unsigned int sz) {
    delete[] ptr;
}

// Instantiate WorkspaceBuffer for the supported types
template class WorkspaceBuffer<mpfr_t>;
template class WorkspaceBuffer<mpz_t>;
template class WorkspaceBuffer<int64_t>;
template class WorkspaceBuffer<double>;

}