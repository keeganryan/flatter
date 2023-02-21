#include "irregular.h"

#include <cassert>

#include "problems/lattice_reduction.h"
#include "problems/matrix_multiplication.h"
#include "problems/qr_factorization.h"
#include "problems/size_reduction.h"
#include "problems/fused_qr_size_reduction.h"

#include "recursive_generic.h"
#include "sublattice_split_3.h"
#include "sublattice_split_2.h"

namespace flatter {
namespace LatticeReductionImpl {

const std::string Irregular::impl_name() {return "Irregular";}

Irregular::Irregular(const LatticeReductionParams& p, const ComputationContext& cc) :
    Base(p, cc)
{
    _is_configured = false;
    configure(p, cc);
}

Irregular::~Irregular() {
    if (_is_configured) {
        unconfigure();
    }
}

void Irregular::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Irregular::configure(const LatticeReductionParams& p, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(p, cc);

    assert(m >= n);
    assert(M.is_transposed() == false);
    assert(U.is_transposed() == false);
    assert(M.type() == ElementType::MPZ || M.type() == ElementType::INT64);
    assert(U.type() == ElementType::MPZ || U.type() == ElementType::INT64);

    _is_configured = true;
}

void Irregular::solve() {
    log_start();

    mon->profile_reset(n);

    char* logfile = getenv("FLATTER_AGGRESSIVE_PREC");
    if (logfile != nullptr) {
        this->params.aggressive_precision = true;
    }

    if (!solve_triangular()) {
        solve_rectangular();
    }

    log_end();
}

bool Irregular::is_corner_zero(bool left, bool top) {
    MatrixData<mpz_t> dM = M.data<mpz_t>();

    for (unsigned int i = 0; i < m; i++) {
        // first row is one where 0 entries are zero
        unsigned int row = (top) ? m - i - 1 : i;
        for (unsigned int j = 0; j <= i; j++) {
            unsigned int col = (left) ? j : n - j - 1;
            if (j < i) {
                if (mpz_cmp_ui(dM(row, col), 0) != 0) {
                    return false;
                }
            } else {
                if (mpz_cmp_ui(dM(row, col), 0) == 0) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool Irregular::is_triangular(bool& flip_rows, bool& flip_cols) {
    if (is_corner_zero(true, false)) {
        // bottom left is zero
        flip_rows = false;
        flip_cols = false;
        return true;
    } else if (is_corner_zero(true, true)) {
        // top left is zero
        flip_rows = true;
        flip_cols = false;
        return true;
    } else if (is_corner_zero(false, false)) {
        // bottom right is zero
        flip_rows = false;
        flip_cols = true;
        return true;
    } else if (is_corner_zero(false, true)) {
        // top right is zero
        flip_rows = true;
        flip_cols = true;
        return true;
    }
    return false;
}

void Irregular::flip_mat(Matrix& M, bool flip_rows, bool flip_cols) {
    MatrixData<mpz_t> dM = M.data<mpz_t>();

    if (flip_rows) {
        for (unsigned int i = 0; i < m / 2; i++) {
            for (unsigned int j = 0; j < n; j++) {
                mpz_swap(
                    dM(i, j),
                    dM(m - i - 1, j)
                );
            }
        }
    }
    if (flip_cols) {
        for (unsigned int i = 0; i < m; i++) {
            for (unsigned int j = 0; j < n / 2; j++) {
                mpz_swap(
                    dM(i, j),
                    dM(i, n - j - 1)
                );
            }
        }
    }
}

bool Irregular::solve_triangular() {
    // Try to solve as triangular system

    if (m != n) {
        return false;
    }
    bool flip_rows, flip_cols;
    if (!is_triangular(flip_rows, flip_cols)) {
        return false;
    }
    flip_mat(M, flip_rows, flip_cols);
    assert(M.is_upper_triangular());

    Matrix U_sr(ElementType::MPZ, n, n);
    MatrixMultiplication mm(U, U_sr, U, cc);
    SizeReduction szred(M, U_sr, cc);
    szred.solve();

    double* profile = new double[n];
    
    assert(M.type() == ElementType::MPZ);
    double maxv, minv;
    MatrixData<mpz_t> dM = M.data<mpz_t>();
    maxv = minv = mpz_sizeinbase(dM(0,0),2);
    profile[0] = maxv;
    for (unsigned int i = 1; i < n; i++) {
        unsigned int sz = mpz_sizeinbase(dM(i,i),2);
        maxv = std::max(maxv, (double)sz);
        minv = std::min(minv, (double)sz);
        long int exp;
        double d = mpz_get_d_2exp(&exp, dM(i,i));
        profile[i] = log2(fabs(d)) + exp;
    }

    Lattice L(M);
    for (unsigned int i = 0; i < n; i++) {
        L.profile[i] = profile[i];
    }
    mon->profile_update(
        &profile[0],
        0, n
    );

    LatticeReductionParams p2(L, U, rhf, true);

    p2.proved = this->params.proved;
    p2.goal = this->params.goal;
    p2.B2 = this->params.B2;
    p2.U2 = this->params.U2;
    p2.log_cond = this->params.log_cond;
    p2.aggressive_precision = this->params.aggressive_precision;

    p2.phase = 2;

    p2.profile_offset = new double[n];
    for (unsigned int i = 0; i < n; i++) {
        p2.profile_offset[i] = 0;
    }

    p2.split = new SubSplitPhase2(n);

    LatticeReduction latred(p2, cc);
    latred.solve();

    for (unsigned int i = 0; i < n; i++) {
        params.L.profile[i] = p2.L.profile[i];
    }

    delete p2.split;

    mm.solve();

    if (params.B2.nrows() > 0) {
        MatrixMultiplication mm2 (params.U2, U_sr, params.U2, cc);
        mm2.solve();
    }

    delete[] profile;
    delete[] p2.profile_offset;

    flip_mat(M, flip_rows, false);
    flip_mat(U, flip_cols, false);

    return true;
}

void Irregular::solve_rectangular() {
    assert(M.type() == ElementType::MPZ);

    Lattice L(M);

    LatticeReductionParams p(L, U, rhf, true);
    p.proved = this->params.proved;
    p.goal = this->params.goal;
    p.phase = 1;
    if (this->params.log_cond > 0) {
        p.log_cond = this->params.log_cond;
    } else {
        // Optimistic, the actual bound is O(n * log|M|),
        // but this is good enough in practice
        p.log_cond = M.prec() + 2*n;
    }
    
    p.profile_offset = new double[n];
    for (unsigned int i = 0; i < n; i++) {
        p.profile_offset[i] = 0;
    }
    p.split = new SubSplitPhase2(n);
    p.B2 = this->params.B2;
    p.U2 = this->params.U2;
    p.aggressive_precision = this->params.aggressive_precision;

    LatticeReduction latred(p, cc);
    latred.solve();

    for (unsigned int i = 0; i < n; i++) {
        params.L.profile[i] = p.L.profile[i];
    }

    delete p.split;
    delete[] p.profile_offset;
}

}
}