#include "monitor.h"

#include <cassert>
#include <climits>
#include <cstdarg>

#include <iostream>
#include <omp.h>

namespace flatter {

char Monitor::logline_buf[LOGLINE_LEN];
unsigned int Monitor::logbuf_offs;
unsigned int Monitor::current_prob_id = 0;
std::mutex Monitor::problems_lock;
unsigned int Monitor::first_id = 0;

Monitor& Monitor::getInstance() {
    static Monitor mon;
    return mon;
}

Monitor::Monitor () {
    gen = std::mt19937(rd());
    prob_id_dist = std::uniform_int_distribution<unsigned int>(0, UINT_MAX);
    has_logfile = false;
}

Monitor::~Monitor() {
}

void Monitor::set_logfile(const std::string& fname) {
    stop();
    
    f = fopen(fname.c_str(), "w+");
    has_logfile = true;

    logbuf_offs = 0;
    unsigned int t = time(NULL);
    log("Start %u\n", t);
    log_commit();
}

void Monitor::stop() {
    if (!has_logfile) {
        return;
    }
    fclose(f);
    f = nullptr;
    has_logfile = false;
}

void Monitor::profile_reset(unsigned int dim) {
    cur_dim = dim;
    reduction_start = omp_get_wtime();
    log("profile(%u)\n", dim);
    log_commit();
}

void Monitor::profile_update(double* profile, unsigned int start, unsigned int end) {
    profile_update(profile, nullptr, start, end);
}

void Monitor::profile_update(double* profile, double* global_offsets, unsigned int start, unsigned int end) {
    auto subdim = end - start;
    if (subdim < cur_dim / 20) {
        return;
    }
    ts cur = omp_get_wtime();
    log("profile(%u,%u)[%f] ", start, end, cur-reduction_start);
    for (unsigned int i = 0; i < end - start; i++) {
        double offset = 0;
        if (global_offsets != nullptr) {
            offset = global_offsets[i];
        }
        log("%0.2f+%0.2f ", profile[i], offset);
    }
    log("\n");
    log_commit();
}

void Monitor::precision_update(unsigned int prec, unsigned int start, unsigned int end) {
    log("Setting precision to %u\n", prec);
    log_commit();
}

void Monitor::start_problem(const std::string& prob, const std::string& impl,
                        const std::string& header, const std::string& params,
                        const ComputationContext& cc) {
    if (!has_logfile) {
        return;
    }

    timer_prob tp;
    tp.params_str = params;
    // Get unique problem label
    std::string key = prob + impl;

    #pragma omp critical
    {
        if (problem_labels.find(key) == problem_labels.end()) {
            // Generate lbl
            unsigned int lbl = prob_id_dist(rd);
            problem_labels.insert(std::pair<const std::string, unsigned int>(key, lbl));
            // Register label
            log("R %08x |%s|%s|%s|\n", lbl, prob.c_str(), impl.c_str(), header.c_str());
            log_commit(true);
        }
        tp.label = problem_labels.find(key)->second;
    }
    
    tp.prob_id = prob_id_dist(rd);
    tp.parent_id = current_prob_id;
    tp.start = omp_get_wtime();

    problems_lock.lock();
    problems.insert(
        std::pair<unsigned int, timer_prob>(tp.prob_id, tp)
    );
    if (first_id == 0) {
        first_id = tp.prob_id;
    }
    problems_lock.unlock();

    current_prob_id = tp.prob_id;
    assert(current_prob_id != 0);
}

void Monitor::end_problem(const ComputationContext& cc) {
    if (!has_logfile) {
        return;
    }

    assert(current_prob_id != 0);
    ts end = omp_get_wtime();
    timer_prob tp, first;

    problems_lock.lock();
    auto elem = problems.find(current_prob_id);
    
    assert(elem != problems.end());
    tp = elem->second;
    problems.erase(elem);
    if (first_id == tp.prob_id) {
        first_id = 0;
        first = tp;
    } else {
        first = problems.find(first_id)->second;
    }
    problems_lock.unlock();

    int nthreads = cc.nthreads();
    tp.end = end;
    tp.duration = tp.end-tp.start;
    first.duration = tp.end - first.start;

    if (tp.duration > 0.01 * first.duration) {
        log("T ");
        // Print own type and ID
        log("%08x %08x ", tp.label, tp.prob_id);
        // Print parent ID
        unsigned int parent_id = tp.parent_id;
        log("%08x", parent_id);
        log(" %u", 1);
        log(" %u", nthreads);
        log(" {%s}", tp.params_str.c_str());
        log(" %f\n", tp.duration);
        log_commit();
    }

    current_prob_id = tp.parent_id;
}


void Monitor::log(const char* s, ...) {
    if (!has_logfile) {
        return;
    }
    va_list args;
    va_start(args, s);
    logbuf_offs += vsnprintf(logline_buf + logbuf_offs, LOGLINE_LEN - logbuf_offs, s, args);
    assert(logbuf_offs < LOGLINE_LEN - 1);
    va_end(args);
}

void Monitor::log_commit(bool in_critical) {
    if (!has_logfile) {
        return;
    }
    if (!in_critical) {
        #pragma omp critical
        {
            fwrite(logline_buf, 1, logbuf_offs, f);
            fflush(f);
            logbuf_offs = 0;
        }
    } else {
        fwrite(logline_buf, 1, logbuf_offs, f);
        fflush(f);
        logbuf_offs = 0;
    }
}

}