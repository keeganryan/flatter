#pragma once

#include <string>

namespace flatter {

class ComputationContext {
public:
    ComputationContext();
    ComputationContext(unsigned int max_threads);

    bool is_threaded() const;
    unsigned int nthreads() const;

    void barrier() const;
    
private:
    unsigned int max_threads_;
};

}