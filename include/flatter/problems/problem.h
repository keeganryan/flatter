#pragma once

#include <string>

#include <flatter/computation_context.h>
#include <flatter/monitor.h>

using namespace flatter;

namespace flatter {

class Params {
public:
    unsigned int val1;
};

class Problem {
public:
    Problem();

    void configure();
    virtual void solve() = 0;

    const ComputationContext* get_computation_context();

    virtual const std::string prob_name();
    virtual const std::string impl_name();
    virtual const std::string param_headers();
    virtual std::string get_param_values();

protected:
    void log_start();
    void log_end();

    Monitor* mon;
    ComputationContext cc;
};

}