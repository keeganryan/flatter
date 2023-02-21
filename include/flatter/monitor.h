#pragma once

#include <map>
#include <chrono>
#include <random>
#include <vector>
#include <memory>
#include <mutex>

#include <omp.h>

#include "computation_context.h"

#define LOGLINE_LEN (16384<<4)

namespace flatter {

typedef double ts;

struct timer_prob {
    char type;
    ts start;
    ts end;
    std::vector<unsigned int> params;
    std::string params_str;
    double duration;
    unsigned int prob_id;
    unsigned int parent_id;
    unsigned int label;
};

class Monitor {
public:
    static Monitor& getInstance();
    Monitor(Monitor const&) = delete;
    void operator=(Monitor const&) = delete;

    void set_logfile(const std::string& fname);

    void stop();

    void start_problem(const std::string& prob, const std::string& impl,
                       const std::string& header, const std::string& params,
                       const ComputationContext& cc);
    void end_problem(const ComputationContext& cc);

    void profile_reset(unsigned int dim);
    void profile_update(double* profile, unsigned int start, unsigned int end);
    void profile_update(double* profile, double* global_offsets, unsigned int start, unsigned int end);
    void precision_update(unsigned int pre, unsigned int start, unsigned int end);

private:
    Monitor();
    ~Monitor();

    void log(const char* s, ...);
    void log_commit(bool in_critical=false);

    std::map<const std::string, unsigned int> problem_labels;
    static std::mutex problems_lock;
    std::map<unsigned int, timer_prob> problems;

    static char logline_buf[LOGLINE_LEN];
    #pragma omp threadprivate(logline_buf)

    static unsigned int logbuf_offs;
    #pragma omp threadprivate(logbuf_offs)

    static unsigned int first_id;
    static unsigned int current_prob_id;
    #pragma omp threadprivate(current_prob_id)

    ComputationContext cc;
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<unsigned int> prob_id_dist;
    ts reduction_start;

    unsigned int cur_dim;

    FILE* f;
    bool has_logfile;
};

}