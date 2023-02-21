#pragma once

#include "recursive_generic.h"

namespace flatter {
namespace LatticeReductionImpl {

struct Tile {
    unsigned int start;
    unsigned int end;
    bool reduce;
};

class Heuristic3 : public RecursiveGeneric {
public:
    Heuristic3(const LatticeReductionParams& p, const ComputationContext& cc);

    const std::string impl_name();

protected:
    virtual void do_checks();

    virtual void init_solver();
    virtual void fini_solver();
    
    virtual bool is_reduced();
    virtual void setup_sublattice_reductions();
    virtual void reduce_sublattices();

    virtual void update_representation();
    virtual void collect_U();
    virtual void final_sr();

    virtual void init_iter();
    virtual void fini_iter();

    virtual void set_profile();
    virtual void compress_R();
    virtual void set_precision(unsigned int prec);

    virtual void init_compressed_B();
    virtual unsigned int get_precision_from_spread(double spread);

    void init_tiles();
    void fini_tiles();
    Matrix get_tile(Matrix M, unsigned int row, unsigned int col);
    void lr(unsigned int s_ind);
    void update_b(unsigned int row, unsigned int i);
    void update_b_next(Matrix U_i, unsigned int row, unsigned int i, unsigned int j);
    void qr(unsigned int i);
    void sr(Matrix U_i, unsigned int row, unsigned int col);
    void update_u(Matrix U_i, unsigned int row, unsigned int i, unsigned int j);

    Matrix U_sr;
    Matrix U_tmp;
    Matrix tau;
    std::vector<std::pair<unsigned int, unsigned int>> U_mul_inds;
    std::vector<Tile> tiles;
    std::vector<unsigned int> tiles_to_reduce;
    unsigned int num_tiles;
    Profile profile_next;
    bool aggressive_precision;
};

}
}