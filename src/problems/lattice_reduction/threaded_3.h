#pragma once

#include "heuristic_3.h"

namespace flatter {
namespace LatticeReductionImpl {

class Threaded3 : public Heuristic3 {
public:
    Threaded3(const LatticeReductionParams& p, const ComputationContext& cc);

    const std::string impl_name();

    //void solve(void);
protected:
    virtual void init_iter();
    virtual void fini_iter();
    void reduce_and_update();
    
    void main_loop();
    void solve();

    void lr(unsigned int s_ind);
    bool use_tasks;
    void setup_taskvars();
    void cleanup_taskvars();
    unsigned int tvi(char c, unsigned int i, unsigned int j);
    char* taskvars; //!< Here to provide synchronization between OpenMP tasks
};

}
}