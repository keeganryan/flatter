#pragma once

#include "problems/lattice_reduction/base.h"

namespace flatter {
namespace LatticeReductionImpl {

/**
 * Irregular Lattice reduction used when input basis is rectangular or not upper triangular.
 */
class Irregular : public Base {
public:
    Irregular(const LatticeReductionParams& p, const ComputationContext& cc);
    ~Irregular();

    const std::string impl_name();

    void configure(const LatticeReductionParams& p, const ComputationContext& cc);
    void solve(void);

private:
    void unconfigure();

    bool is_corner_zero(bool left, bool top);
    bool is_triangular(bool& flip_rows, bool& flip_cols);

    void flip_mat(Matrix& M, bool flip_rows, bool flip_cols);

    bool solve_triangular();
    void solve_rectangular();

    bool _is_configured;
};

}
}