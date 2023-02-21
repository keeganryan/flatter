#pragma once

#include "problems/size_reduction/base.h"
#include "workspace_buffer.h"

namespace flatter {
namespace SizeReductionImpl {

class Blocked : public Base {
public:
    Blocked(const Matrix& R, const Matrix& U,
         const ComputationContext& cc);
    ~Blocked();

    const std::string impl_name();

    void configure(const Matrix& R, const Matrix& U,
         const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    Matrix get_tile(Matrix& M, unsigned int i, unsigned int j);


    void update_diagonal(unsigned int diag);

    void diag_diag(unsigned int i);
    void diag_above(unsigned int b_i, unsigned int b_j);
    void inner_inner(unsigned int b_i, unsigned int b_j);
    void inner_above(unsigned int b_i, unsigned int b_j_l, unsigned int b_j_r);

    bool _is_configured;

    mpfr_rnd_t rnd;
    unsigned int nb_c;
    unsigned int nb_r;
    unsigned int bs;

    WorkspaceBuffer<mpz_t>* wsb;
    Matrix T;
    Matrix W;
};

}
}