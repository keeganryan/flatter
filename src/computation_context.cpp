#include "computation_context.h"

#include <cassert>
#include <omp.h>
#include <iostream>

namespace flatter {

ComputationContext::ComputationContext() {
    max_threads_ = omp_get_max_threads();
}

ComputationContext::ComputationContext(unsigned int max_threads) {
    max_threads_ = max_threads;
}

bool ComputationContext::is_threaded() const {
    return max_threads_ > 1;
}

unsigned int ComputationContext::nthreads() const {
    return max_threads_;
}

void ComputationContext::barrier() const {
    return;
}

}