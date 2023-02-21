#pragma once

#include "problems/lattice_reduction/base.h"

namespace flatter {
namespace LatticeReductionImpl {

class RecursiveGeneric : public Base {
public:
    RecursiveGeneric(const LatticeReductionParams& p, const ComputationContext& cc);
    ~RecursiveGeneric();

    const std::string impl_name();

    void configure(const LatticeReductionParams& p, const ComputationContext& cc);
    void solve(void);

protected:
    virtual void do_checks();

    virtual void init_solver();
    virtual void fini_solver();

    virtual void init_iter();
    virtual void fini_iter();

    virtual bool is_reduced(); 

    virtual void setup_sublattice_reductions();
    virtual void cleanup_sublattice_reductions();
    virtual void reduce_sublattices();
    virtual void update_representation();
    virtual void collect_U();
    virtual void final_sr();

    virtual void set_profile();
    virtual unsigned int get_shifts_for_compression(int* shifts);
    virtual void compress_R();
    virtual void set_precision(unsigned int prec);

    void unconfigure();

    void log_profile();

    unsigned int get_initial_precision();
    virtual void init_compressed_B();
    virtual unsigned int get_precision_from_spread(double spread);

    Profile profile; //!< local copy of profile
    double* local_profile_offsets; //!< How we scaled the profile
    double* global_profile_offsets; //!< How we (globally) scaled the profile
    Matrix B; //!< Internal representation of basis
    Matrix R; //!< Internal floating point basis representation
    Matrix B_next; //!< Next internal representation of basis

    unsigned int original_precision;
    unsigned int precision;
    mpfr_rnd_t rnd;

    unsigned int iterations;
    std::vector<Matrix> U_iters;
    std::vector<int*> compression_iters;
    bool lattice_changed;

    unsigned int num_sublattices;
    std::vector<std::pair<unsigned int, unsigned int>> sublattice_inds;
    std::vector<LatticeReductionParams> sub_params;
    std::vector<Lattice> L_subs;
    std::vector<Matrix> U_subs;

    double logdet;

    bool _is_configured;
};

}
}