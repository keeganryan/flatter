#include "problems/problem.h"

namespace flatter {

const std::string Problem::prob_name() {return "Generic Problem";}
const std::string Problem::impl_name() {return "Base Implementation";}
const std::string Problem::param_headers() {return "";}
std::string Problem::get_param_values() {
    return "";
}

Problem::Problem() {
    mon = &Monitor::getInstance();
}

const ComputationContext* Problem::get_computation_context() {
    return &cc;
}

void Problem::log_start() {
    mon->start_problem(this->prob_name(), this->impl_name(),
                        this->param_headers(), this->get_param_values(),
                        cc);
}

void Problem::log_end() {
    mon->end_problem(cc);
}

}