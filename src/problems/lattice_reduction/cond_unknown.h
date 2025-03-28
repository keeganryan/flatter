#pragma once

#include "problems/lattice_reduction/base.h"

namespace flatter {
namespace LatticeReductionImpl {

class CondUnknown : public Base {
public:
    CondUnknown(const LatticeReductionParams& p, const ComputationContext& cc);
    ~CondUnknown();

    const std::string impl_name();

    void configure(const LatticeReductionParams& p, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    void apply_U(const Matrix& U1, const Matrix& U2);
    void apply_perm(const Matrix& U);
    void extract_similar(const Matrix& B, unsigned int prec, Matrix Bsim, Matrix U, unsigned int& num_valid, int& shift_amount, double& spread);
    void sort_by_size(Matrix B);
    bool refine_basis();

    bool _is_configured;

    Matrix B;
    Matrix R;
    unsigned int max_rank;
    unsigned int working_prec;
};

}
}