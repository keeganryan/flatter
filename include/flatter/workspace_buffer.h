#pragma once

#include <vector>
#include <memory>

namespace flatter {

/**
 * Simple memory allocation of a templated type
 */
template <class T>
class WorkspaceBuffer {
public:
    WorkspaceBuffer(unsigned int sz, unsigned int prec);

    void set_precision(unsigned int prec);
    T* walloc(unsigned int sz);
    void wfree(T* ptr, unsigned int sz);

private:
    static void ws_delete(T*, unsigned int sz);

    unsigned int prec_;
    std::shared_ptr<T[]> ws_;
    unsigned int sz_;
    unsigned int offset_;
};

}