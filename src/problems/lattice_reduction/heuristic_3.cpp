#include "heuristic_3.h"

#include "problems/fused_qr_size_reduction.h"
#include "problems/lattice_reduction.h"
#include "problems/matrix_multiplication.h"
#include "problems/qr_factorization.h"
#include "problems/relative_size_reduction.h"

#include <cassert>

namespace flatter {
namespace LatticeReductionImpl {

const std::string Heuristic3::impl_name() {return "Heuristic3";}

Heuristic3::Heuristic3(const LatticeReductionParams& p, const ComputationContext& cc) :
    RecursiveGeneric(p, cc)
{}

void Heuristic3::do_checks() {
    assert(params.split != nullptr);
}

void Heuristic3::init_compressed_B() {
    MatrixData<mpz_t> dM = this->M.data<mpz_t>();
    for (unsigned int i = 0; i < n; i++) {
        long int exp;
        double d = mpz_get_d_2exp(&exp, dM(i,i));
        double newval = log2(fabs(d)) + exp;
        profile[i] = newval;
    }

    log_profile();

    int *shifts = new int[n];
    compression_iters.push_back(shifts);
    
    unsigned int precision = this->get_shifts_for_compression(shifts);

    MatrixData<mpz_t> dB = B.data<mpz_t>();
    for (unsigned int j = 0; j < n; j++) {
        local_profile_offsets[j] += shifts[j];
        global_profile_offsets[j] += shifts[j];
        profile[j] -= shifts[j];

        for (unsigned int i = 0; i < n; i++) {
            if (i > j) {
                mpz_set_ui(dB(i,j), 0);
            } else {
                if (shifts[j] < 0) {
                    mpz_mul_2exp(dB(i,j), dM(i,j), -shifts[j]);
                } else {
                    mpz_div_2exp(dB(i,j), dM(i,j), shifts[j]);
                }
            }
        }
    }

    set_precision(precision);
    Matrix::copy(R, B);
}

unsigned int Heuristic3::get_precision_from_spread(double spread) {
    unsigned int precision;
    // This is enough precision, but sometimes we can do even better.
    if (this->params.aggressive_precision) {
        precision = spread + 30;
    } else {
        precision = 2.0 * spread + 30 + 2*n;
    }
    return precision;
}

bool Heuristic3::is_reduced() {
    if (!this->params.split->stopping_point()) {
        return false;
    }
    if (this->params.split->stopping_point() && !lattice_changed) {
        return true;
    }

    return params.goal.check(profile);
}

void Heuristic3::init_tiles() {
    Tile t;
    unsigned int last_ind = 0;
    for (unsigned int i = 0; i < num_sublattices; i++) {
        auto s = sublattice_inds[i];
        unsigned int start = s.first;
        unsigned int end = s.second;

        if (start > last_ind) {
            t.start = last_ind;
            t.end = start;
            t.reduce = false;
            tiles.push_back(t);
        }

        tiles_to_reduce.push_back(tiles.size());
        t.start = start;
        t.end = end;
        t.reduce = true;
        tiles.push_back(t);
        last_ind = end;
    }
    if (last_ind < n) {
        t.start = last_ind;
        t.end = n;
        t.reduce = false;
        tiles.push_back(t);
    }
    num_tiles = tiles.size();
}

void Heuristic3::fini_tiles() {
    tiles.clear();
    tiles_to_reduce.clear();
}

void Heuristic3::setup_sublattice_reductions() {
    // Set the indices of the sublattice to reduce.
    if (this->params.split->stopping_point()) {
        lattice_changed = false;
    }
    
    auto sublats = this->params.split->get_sublattices();
    num_sublattices = sublats.size();

    unsigned int min_start = n;
    unsigned int max_end = 0;

    for (unsigned int i = 0; i < num_sublattices; i++) {
        sublattice s = sublats[i];

        unsigned int start, end;
        start = s.start;
        end = s.end;
        min_start = std::min(min_start, start);
        max_end = std::max(max_end, end);

        sublattice_inds.push_back(
            std::make_pair(start, end)
        );

        Matrix B_sub(ElementType::MPZ, end-start, end-start);
        
        Lattice L_sub (B_sub);
        L_sub.profile = profile.subprofile(start, end);

        L_subs.push_back(L_sub);
        Matrix::copy(B_sub, B.submatrix(start, end, start, end));

        Matrix U_sub(ElementType::MPZ, end-start, end-start);
        U_subs.push_back(U_sub);

        LatticeReductionParams params(L_sub, U_sub, rhf, true);

        LatticeReductionGoal goal1 = this->params.goal.subgoal(start, end);
        double current_drop = profile.get_drop();
        LatticeReductionGoal goal2 = LatticeReductionGoal::from_drop(
            end-start, current_drop / 6);

        if (goal1.get_slope() < goal2.get_slope()) {
            params.goal = goal2;
        } else {
            params.goal = goal1;
        }
        params.split = this->params.split->get_child_split(i);
        params.proved = this->params.proved;
        params.offset = this->offset + start;
        params.profile_offset = &global_profile_offsets[start];
        params.lvalid = n / 4;
        params.rvalid = n / 4;
        params.phase = this->params.phase;
        params.aggressive_precision = this->params.aggressive_precision;

        sub_params.push_back(params);
    }

    U_mul_inds.push_back(std::make_pair(min_start, max_end));
}

void Heuristic3::reduce_sublattices() {
    for (unsigned int i = 0; i < n; i++) {
        profile_next[i] = profile[i];
    }
    for (unsigned int i = 0; i < num_sublattices; i++) {
        lr(i);
    }
}

void Heuristic3::update_representation() {
    // Assumes that U_subs, sublattice information is all up to date
    // Incorporates the sublattice reduction information into the
    // current state.
    Matrix U_i = U_iters.back();
    U_i.set_identity();
    U_sr.set_identity();

    Matrix::copy(B_next, B);

    // Update U_i
    for (unsigned int i = 0; i < num_sublattices; i++) {
        unsigned int ind = tiles_to_reduce[i];
        Matrix::copy(
            get_tile(U_i, ind, ind),
            U_subs[i]
        );
    }

    // Update B_next.
    // B_next will not necessarily be size reduced or triangular.

    for (unsigned int i = 0; i < num_tiles; i++) {
        for (unsigned int j = 0; j <= i; j++) {
            update_b(j, i);
        }
    }

    // Reduce precision
    double spread = profile_next.get_spread();
    unsigned int precision = get_precision_from_spread(spread);
    if (precision > 53 && R.type() == ElementType::DOUBLE) {
        // Could happen in final stages when doing last FQSR
        Matrix new_R(ElementType::MPFR, m, n, prec);
        Matrix new_tau(ElementType::MPFR, n, 1, prec);
        Matrix::copy(new_R, R);
        Matrix::copy(new_tau, tau);
        this->R = new_R;
        this->tau = new_tau;
    }
    R.set_precision(this->precision);
    tau.set_precision(this->precision);
    this->precision = precision;

    // QR factorize
    for (unsigned int i = 0; i < num_tiles; i++) {
        qr(i);
    }

    for (unsigned int i = 0; i < num_tiles; i++) {
        unsigned int tile_col = i;

        for (unsigned int j = 0; j < tile_col; j++) {
            // Work from bottom up
            unsigned int tile_row = tile_col - j - 1;

            sr(U_sr, tile_row, tile_col);

            for (unsigned int k = 0; k < tile_row; k++) {
                // Update B and U_sr
                update_b_next(U_sr, k, tile_row, tile_col);
            }
            for (unsigned int k = 0; k <= tile_row; k++) {
                // Update B and U_sr
                update_u(U_sr, k, tile_row, tile_col);
            }

            Matrix::copy(B, B_next);
        }
    }
    
    // Use R-factor to compute profile
    set_profile();
    // Take R-factor, compress, and extract compressed basis to B.
    compress_R();
    Matrix::copy(B, R);

    assert(B.is_upper_triangular());
}

void Heuristic3::collect_U() {
    assert(U_iters.size() == compression_iters.size() - 1);

    delete[] compression_iters.back();
    compression_iters.pop_back();

    // Build U by multiplying all U_iters, starting from the end
    if (U_iters.size() == 0) {
        U.set_identity();
        assert(compression_iters.size() == 0);
        return;
    }

    U_tmp = Matrix(ElementType::MPZ, n, n);

    Matrix U_iter = U_iters.back();
    MatrixData<mpz_t> dU_iter = U_iter.data<mpz_t>();
    // Do D^-1 UD
    int* D = compression_iters.back();
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if (D[i] == D[j]) {
                // Do nothing
            } else if (D[i] > D[j]) {
                // then we should be below the diagonal and zero
                assert(mpz_cmp_ui(dU_iter(i,j), 0) == 0);
            } else {
                // Then we should be above the diagonal and need to shift
                mpz_mul_2exp(dU_iter(i,j), dU_iter(i,j), D[j] - D[i]);
            }
        }
    }
    delete[] compression_iters.back();
    compression_iters.pop_back();

    Matrix::copy(U, U_iter);
    U_iters.pop_back();
    U_mul_inds.pop_back();

    while (U_iters.size() > 0) {
        auto bounds = U_mul_inds.back();
        unsigned int start = bounds.first;
        unsigned int end = bounds.second;

        Matrix U_iter = U_iters.back();
        int* D = compression_iters.back();

        MatrixData<mpz_t> dU_iter = U_iter.data<mpz_t>();
        // Do D^-1 UD
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < n; j++) {
                if (D[i] == D[j]) {
                    // Do nothing
                } else if (D[i] > D[j]) {
                    // then we should be below the diagonal and zero
                    assert(mpz_cmp_ui(dU_iter(i,j), 0) == 0);
                } else {
                    // Then we should be above the diagonal and need to shift
                    mpz_mul_2exp(dU_iter(i,j), dU_iter(i,j), D[j] - D[i]);
                }
            }
        }

        // Take advantage of the fact that U_iter is structured as
        // [I A B]
        // [0 C D]
        // [0 0 I]
        MatrixMultiplication ui_mm (
            U_tmp.submatrix(0, end, 0, n),
            U_iter.submatrix(0, end, start, n),
            U.submatrix(start, n, 0, n),
            cc
        );
        ui_mm.solve();

        MatrixData<mpz_t> dU_tmp = U_tmp.data<mpz_t>();
        MatrixData<mpz_t> dU = U.data<mpz_t>();
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < n; j++) {
                if (i < start) {
                    // Accumulate
                    mpz_add(dU(i,j), dU(i,j), dU_tmp(i, j));
                } else if (i < end) {
                    // Do not accumulate
                    mpz_set(dU(i, j), dU_tmp(i, j));
                } else {
                    // U does not change
                }
            }
        }
        //Matrix::copy(U, U_tmp);
        U_iters.pop_back();
        U_mul_inds.pop_back();

        delete[] compression_iters.back();
        compression_iters.pop_back();
    }

    assert(compression_iters.size() == 0);
}

void Heuristic3::final_sr() {
    RecursiveGeneric::final_sr();
}

void Heuristic3::set_profile() {
    if (R.type() == ElementType::MPFR) {
        RecursiveGeneric::set_profile();
        return;
    }
    
    // Assumes this->R is up to date and upper triangular
    // Writes to profile array with current profile
    MatrixData<double> dR = R.data<double>();
    for (unsigned int i = 0; i < n; i++) {
        double d = dR(i, i);
        double newval = log2(fabs(d));
        if (profile[i] != newval) {
            lattice_changed = true;
        }
        profile[i] = newval;
    }
}

void Heuristic3::compress_R() {
    if (R.type() == ElementType::MPFR) {
        RecursiveGeneric::compress_R();
        return;
    }

    // Assumes this->R is upper triangular and profile is valid
    // Writes compressed basis to B and updates profile.

    // Calculate how many bits are needed
    double *max_from_left = new double[n];
    double *min_from_right = new double[n];
    int *shifts = compression_iters.back();

    max_from_left[0] = profile[0];
    min_from_right[n - 1] = profile[n - 1];
    for (unsigned int i = 0; i < n - 1; i++) {
        max_from_left[i + 1] = std::max(profile[i + 1], max_from_left[i]);
        min_from_right[n - i - 2] = std::min(profile[n - i - 2], min_from_right[n - i - 1]);
    }

    shifts[0] = 0;
    for (unsigned int i = 1; i < n; i++) {
        shifts[i] = shifts[i - 1];

        double compress_amount = min_from_right[i] - max_from_left[i - 1];
        if (compress_amount <= 1) {
            continue;
        }

        compress_amount = floor(compress_amount - 1);

        shifts[i] += compress_amount;
    }

    // Ignore compression bit for now
    double spread = max_from_left[n - 1] - shifts[n-1] - min_from_right[0];

    unsigned int precision = get_precision_from_spread(spread);

    // Do scaling operation and copy from R to B
    int new_shift = ceil(max_from_left[n-1]) - shifts[n-1] - (int)precision;

    MatrixData<double> dR = R.data<double>();
    for (unsigned int j = 0; j < n; j++) {
        local_profile_offsets[j] += new_shift + shifts[j];
        global_profile_offsets[j] += new_shift + shifts[j];
        profile[j] -= new_shift + shifts[j];

        for (unsigned int i = 0; i < n; i++) {
            if (i > j) {
                dR(i,j) = 0;
            } else {
                dR(i, j) *= pow(2, -new_shift - shifts[j]);
            }
        }
    }

    set_precision(precision);

    delete[] max_from_left;
    delete[] min_from_right;
}

void Heuristic3::set_precision(unsigned int prec) {
    this->precision = prec;

    // Do we have to swap out data type for R?
    if (prec <= 53 && R.type() == ElementType::MPFR && false) {
        Matrix new_R(ElementType::DOUBLE, m, n, 53);
        Matrix new_tau(ElementType::DOUBLE, n, 1, 53);
        Matrix::copy(new_R, R);
        Matrix::copy(new_tau, tau);
        this->R = new_R;
        this->tau = new_tau;
    } else if (prec > 53 && R.type() == ElementType::DOUBLE) {
        // Could happen in final stages when doing last FQSR
        Matrix new_R(ElementType::MPFR, m, n, prec);
        Matrix new_tau(ElementType::MPFR, n, 1, prec);
        Matrix::copy(new_R, R);
        Matrix::copy(new_tau, tau);
        this->R = new_R;
        this->tau = new_tau;
    }
    R.set_precision(this->precision);
    tau.set_precision(this->precision);
}

void Heuristic3::init_solver() {
    tau = Matrix(ElementType::MPFR, n, 1, 53);
    U_sr = Matrix(ElementType::MPZ, n, n);
    profile_next = Profile(n);

    RecursiveGeneric::init_solver();

    tiles.clear();
    tiles_to_reduce.clear();
    U_mul_inds.clear();
    aggressive_precision = false;
}

void Heuristic3::fini_solver() {
    RecursiveGeneric::fini_solver();
}

void Heuristic3::init_iter() {
    RecursiveGeneric::init_iter();
    init_tiles();
}

void Heuristic3::fini_iter() {
    fini_tiles();
    this->params.split->advance_sublattices();
    RecursiveGeneric::fini_iter();
}

Matrix Heuristic3::get_tile(Matrix M, unsigned int row, unsigned int col) {
    unsigned int l, r, t, b;
    t = tiles[row].start;
    b = tiles[row].end;
    l = tiles[col].start;
    r = tiles[col].end;
    return M.submatrix(t, b, l, r);
}

void Heuristic3::lr(unsigned int s_ind) {
    LatticeReduction lr(sub_params[s_ind], cc);
    lr.solve();
    auto tile = tiles[tiles_to_reduce[s_ind]];
    auto start = tile.start;
    auto end = tile.end;
    auto params = sub_params[s_ind];
    for (unsigned int i = start; i < end; i++) {

        double newval = params.L.profile[i-start];
        profile_next[i] = newval;
    }

}

void Heuristic3::update_b(unsigned int row, unsigned int i) {
    // Update matrix is based on result of sublattice reduction. We can
    // short circuit if no reduction was done
    // Update tile B(row, j) using B(row, i) and U(i,j)
    Matrix B_next_out, B_out, B_in, U_in;

    unsigned int l, r, t, b;
    l = tiles[i].start;
    r = tiles[i].end;
    t = tiles[row].start;
    b = tiles[row].end;
    
    B_in = B.submatrix(t, b, l, r);
    B_out = B.submatrix(t, b, l, r);
    B_next_out = B_next.submatrix(t, b, l, r);
    Matrix U_i = U_iters.back();
    U_in = U_i.submatrix(l, r, l, r);

    if (!tiles[i].reduce) {
        Matrix::copy(B_next_out, B_out);
        return;
    }
    // B_in is B_out, so no accumulation is necessary
    MatrixMultiplication mm(B_next_out, B_in, U_in, cc);
    mm.solve();
    Matrix::copy(B_out, B_next_out);
}

void Heuristic3::update_b_next(Matrix U_i, unsigned int row, unsigned int i, unsigned int j) {
    assert(i < j);
    // Update tile B(row, j) using B(row, i) and U(i,j)
    Matrix B_out, B_in, U_in;

    B_in = get_tile(B, row, i);
    B_out = get_tile(B_next, row, j);
    U_in = get_tile(U_i, i, j);

    // B_in and B_out are disjoint, so we need to accumulate
    MatrixMultiplication mm(B_out, B_in, U_in, true, cc);
    mm.solve();
}

void Heuristic3::qr(unsigned int i) {
    unsigned int start = tiles[i].start;
    unsigned int end = tiles[i].end;
    Matrix R_sub = R.submatrix(start, end, start, end);
    Matrix tau_sub = tau.submatrix(start, end, 0, 1);

    if (tiles[i].reduce) {

        Matrix::copy(R_sub, B_next.submatrix(start, end, start, end));
        QRFactorization qr_sub(R_sub, tau_sub, cc);
        qr_sub.solve();
    }
}

void Heuristic3::sr(Matrix U_i, unsigned int row, unsigned int col) {
    // Assume the following
    //   R is up to date
    //   tau is up to date
    // Perform the following
    //   Update B_next with integer values, shifted by new_shift
    //   Update U_i with integer values
    //   Update B by performing the same size reduction to the tiles above this one
    //Matrix B_sub = get_tile(B_next, row, row);
    Matrix U_sub = get_tile(U_i, row, col);


    // Take advantage of upper triangular basis
    Matrix B1 = get_tile(B, row, row);
    Matrix B2 = get_tile(B_next, row, col);
    Matrix R_top = get_tile(R, row, col);
    
    RelativeSizeReductionParams params(B1, B2, U_sub);

    params.R2 = R_top;
    if (tiles[row].reduce) {
        params.RV = get_tile(R, row, row);
        unsigned int t = tiles[row].start;
        unsigned int b = tiles[row].end;
        params.tau = tau.submatrix(t, b, 0, 1);
    } else {
        params.is_B1_upper_triangular = true;
    }
    RelativeSizeReduction prob(params, cc);
    prob.solve();
}
void Heuristic3::update_u(Matrix U_i, unsigned int row, unsigned int i, unsigned int j) {
    assert(i < j);
    // Update tile B(row, j) using B(row, i) and U(i,j)
    Matrix U_out = U_iters.back();

    // B_in and B_out are disjoint, so we need to accumulate
    MatrixMultiplication mm(
        get_tile(U_out, row, j),
        get_tile(U_out, row, i),
        get_tile(U_i, i, j),
        true,
        cc
    );
    mm.solve();
}

}
}