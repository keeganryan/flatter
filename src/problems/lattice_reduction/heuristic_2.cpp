#include "heuristic_2.h"

#include "problems/fused_qr_size_reduction.h"
#include "problems/matrix_multiplication.h"
#include "problems/qr_factorization.h"
#include "problems/relative_size_reduction.h"

#include <cassert>

namespace flatter {
namespace LatticeReductionImpl {

const std::string Heuristic2::impl_name() {return "Heuristic2";}

Heuristic2::Heuristic2(const LatticeReductionParams& p, const ComputationContext& cc) :
    Heuristic3(p, cc)
{}

bool Heuristic2::is_reduced() {
    if (iterations == 0 && params.goal.check(profile)) {
        return true;
    }
    
    if (this->params.split->stopping_point()) {
        return true;
    }
    return false;
}

void Heuristic2::setup_sublattice_reductions() {
    // Set the indices of the sublattice to reduce.

    auto sublats = this->params.split->get_sublattices();
    assert(sublats.size() == 1);

    sublattice s = sublats[0];

    unsigned int start, end;
    start = s.start;
    end = s.end;

    sublattice_inds.push_back(
        std::make_pair(start, end)
    );

    num_sublattices = 1;

    Matrix B_sub(ElementType::MPZ, end-start, end-start);
    Lattice L_sub (B_sub);
    L_sub.profile = profile.subprofile(start, end);
    
    L_subs.push_back(L_sub);
    Matrix::copy(B_sub, B.submatrix(start, end, start, end));

    Matrix B2;

    Matrix U_sub(ElementType::MPZ, end-start, end-start);
    U_subs.push_back(U_sub);
    Matrix U2;

    if (start == 0 && end < n) {
        // step L
        unsigned int b2_cols = n - end;
        if (this->params.B2.ncols() != 0) {
            b2_cols += this->params.B2.ncols();
        }
        B2 = Matrix(ElementType::MPZ, end-start, b2_cols);
        U2 = Matrix(ElementType::MPZ, end-start, b2_cols);
        Matrix::copy(B2.submatrix(0, end-start, 0, n-end), B.submatrix(start, end, end, n));
        if (b2_cols > n-end) {
            Matrix::copy(
                B2.submatrix(0, end-start, n-end, b2_cols),
                B2_sim.submatrix(0, end-start, 0, b2_cols - (n-end))
            );
        }
        B2s.push_back(B2);
        U2s.push_back(U2);
    }

    if (start == 0 && end == n && params.B2.ncols() != 0) {
        // step all
        B2 = Matrix(ElementType::MPZ, n, b2_cols);
        U2 = Matrix(ElementType::MPZ, n, b2_cols);
        Matrix::copy(B2.submatrix(0, n, 0, b2_cols), B2_sim);
        B2s.push_back(B2);
        U2s.push_back(U2);
    }

    LatticeReductionParams params(L_sub, U_sub, rhf, true);

    LatticeReductionGoal goal1 = this->params.goal.subgoal(start, end);

    // Get mu_L and mu_R to approximate current_drop
    double mu_L = 0;
    double mu_R = 0;
    for (unsigned int i = 0; i < n/2; i++) {
        mu_L += profile[i];
    }
    for (unsigned int i = n/2; i < n; i++) {
        mu_R += profile[i];
    }
    mu_L /= (n / 2);
    mu_R /= n - (n / 2);

    double current_drop = mu_L - mu_R;
    if (current_drop < 0) {
        current_drop = profile.get_drop();
    }
    LatticeReductionGoal goal2 = LatticeReductionGoal::from_drop(
        end-start, current_drop / 6);

    if (!(start == 0 && end == n) && goal1.get_slope() < goal2.get_slope()) {
        params.goal = goal2;
    } else {
        params.goal = goal1;
    }
    params.split = this->params.split->get_child_split(0);
    assert(params.split != nullptr);
    params.proved = this->params.proved;
    params.offset = this->offset + start;
    params.profile_offset = &global_profile_offsets[start];
    params.lvalid = n / 4;
    params.rvalid = n / 4;
    params.phase = 2;
    params.aggressive_precision = this->params.aggressive_precision;

    params.B2 = B2;
    params.U2 = U2;

    if (start == 0 && end == n) {
        params.phase = 3;
    }

    sub_params.push_back(params);

    U_mul_inds.push_back(std::make_pair(start, end));
}

void Heuristic2::cleanup_sublattice_reductions() {
    RecursiveGeneric::cleanup_sublattice_reductions();
    B2s.clear();
    U2s.clear();
}

void Heuristic2::fini_iter() {
    this->params.split->advance_sublattices();
    RecursiveGeneric::fini_iter();
}

void Heuristic2::init_solver() {
    if (params.B2.ncols() > 0) {
        MatrixData<mpz_t> dU2 = params.U2.data<mpz_t>();
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < params.U2.ncols(); j++) {
                mpz_set_ui(dU2(i, j), 0);
            }
        }
    }

    b2_cols = params.B2.ncols();
    if (b2_cols > 0) {
        B2_orig = Matrix(ElementType::MPZ, m, b2_cols);
        B2_sim = Matrix(ElementType::MPZ, m, b2_cols);
        B_orig = Matrix(ElementType::MPZ, m, n);
        Matrix::copy(B2_orig, params.B2);
        Matrix::copy(B_orig, params.B());
    }

    Heuristic3::init_solver();
}

void Heuristic2::fini_solver() {
    
    collect_U();

    MatrixMultiplication mm(M, M, U, cc);
    mm.solve();

    // Update profile
    for (unsigned int i = 0; i < n; i++) {
        profile[i] += local_profile_offsets[i];
        global_profile_offsets[i] -= local_profile_offsets[i];
        local_profile_offsets[i] = 0;
    }
    for (unsigned int i = 0; i < n; i++) {
        params.L.profile[i] = profile[i];
    }

    delete[] local_profile_offsets;
    delete[] global_profile_offsets;
}

void Heuristic2::update_representation() {
    auto sub_ind = sublattice_inds[0];
    unsigned int start = sub_ind.first;
    unsigned int end = sub_ind.second;
    if (start == 0 && end < n) {
        update_L_representation();
    } else if (start > 0 && end == n) {
        update_R_representation();
    } else {
        update_all_representation();
    }
}

void Heuristic2::init_compressed_B() {

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
    this->get_shifts_for_compression(shifts);
    int total_shift = shifts[0];
    for (unsigned int i = 0; i < n; i++) {
        shifts[i] = total_shift;
    }

    MatrixData<mpz_t> dB = B.data<mpz_t>();
    MatrixData<mpz_t> dB2;
    MatrixData<mpz_t> dB2s;
    if (b2_cols > 0) {
        dB2 = params.B2.data<mpz_t>();
        dB2s = B2_sim.data<mpz_t>();
    }
    for (unsigned int i = 0; i < n; i++) {
        local_profile_offsets[i] += total_shift;
        global_profile_offsets[i] += total_shift;
        profile[i] -= total_shift;
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            if (i > j) {
                mpz_set_ui(dB(i,j), 0);
            } else {
                if (total_shift < 0) {
                    mpz_mul_2exp(dB(i,j), dM(i,j), -total_shift);
                } else {
                    mpz_div_2exp(dB(i,j), dM(i,j), total_shift);
                }
            }
        }

        for (unsigned int j = 0; j < b2_cols; j++) {
            if (total_shift < 0) {
                mpz_mul_2exp(dB2s(i,j), dB2(i,j), -total_shift);
            } else {
                mpz_div_2exp(dB2s(i,j), dB2(i,j), total_shift);
            }
        }

    }
}

void Heuristic2::update_L_representation() {
    assert(num_sublattices == 1);

    auto sub_ind = sublattice_inds[0];
    unsigned int start = sub_ind.first;
    unsigned int end = sub_ind.second;

    // Update profile
    Profile sub_prof = L_subs[0].profile;
    for (unsigned int i = 0; i < end; i++) {
        profile[i] = sub_prof[i];
    }
    
    double spread = profile.get_spread();
    unsigned int precision = get_precision_from_spread(spread);

    // Assumes that U_subs, sublattice information is all up to date
    // Incorporates the sublattice reduction information into the
    // current state.

    Matrix U_i = U_iters.back();
    Matrix U2 = U2s.back();
    Matrix B2 = B2s.back();

    U_i.set_identity();

    Matrix U_sub = U_subs[0];

    // Update U_i
    Matrix::copy(
        U_i.submatrix(start, end, start, end),
        U_sub
    );
    Matrix::copy(U_i.submatrix(start, end, end, n), U2.submatrix(0, end-start, 0, n-end));
    if (b2_cols > 0) {
        Matrix::copy(
            params.U2.submatrix(start, end, 0, b2_cols),
            U2.submatrix(0, end-start, n-end, b2_cols + n-end)
        );
        Matrix::copy(
            B2_sim.submatrix(start, end, 0, b2_cols),
            B2.submatrix(0, end-start, n-end, b2_cols + n-end)
        );
    }

    // Update B_next.
    Matrix::copy(
        B_next.submatrix(start, end, start, end),
        L_subs[0].basis()
    );
    Matrix::copy(
        B_next.submatrix(start, end, end, n),
        B2.submatrix(0, end, 0, n-end)
    );
    Matrix::copy(B_next.submatrix(end, n, end, n), B.submatrix(end, n, end, n));
    
    Matrix R(ElementType::MPFR, end, n + b2_cols, precision);

    Matrix::copy(R.submatrix(0, end, 0, n), B_next.submatrix(0, end, 0, n));
    if (b2_cols > 0) {
        Matrix::copy(R.submatrix(0, end, n, n+b2_cols), B2_sim.submatrix(0, end, 0, b2_cols));
    }
    
    QRFactorization qr(R, cc);
    qr.solve();

    // Use R-factor to compute profile
    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    for (unsigned int i = 0; i < end; i++) {
        long int exp;
        double d = mpfr_get_d_2exp(&exp, dR(i, i), rnd);
        double newval = log2(fabs(d)) + exp;
        profile[i] = newval;
    }
    {
        int *shifts = compression_iters.back();
        this->get_shifts_for_compression(shifts);
        int total_shift = shifts[0];
        for (unsigned int i = 0; i < n; i++) {
            shifts[i] = total_shift;
        }

        MatrixData<mpfr_t> dR = R.data<mpfr_t>();
        MatrixData<mpz_t> dB = B_next.data<mpz_t>();
        MatrixData<mpz_t> dBs;
        if (b2_cols > 0) {
            dBs = B2_sim.data<mpz_t>();
        }
        for (unsigned int i = 0; i < n; i++) {
            local_profile_offsets[i] += total_shift;
            global_profile_offsets[i] += total_shift;
            profile[i] -= total_shift;
        }

        for (unsigned int i = 0; i < end; i++) {
            for (unsigned int j = 0; j < n + b2_cols; j++) {
                if (j < i) {
                    mpfr_set_zero(dR(i,j), 0);
                } else {
                    mpfr_mul_2si(dR(i,j), dR(i,j), -total_shift, rnd);
                }
            }
        }
        for (unsigned int i = end; i < n; i++) {
            for (unsigned int j = 0; j < n; j++) {
                if (j < i) {
                    mpz_set_ui(dB(i,j), 0);
                } else {
                    if (total_shift < 0) {
                        mpz_mul_2exp(dB(i,j), dB(i,j), -total_shift);
                    } else {
                        mpz_div_2exp(dB(i,j), dB(i,j), total_shift);
                    }
                }
            }
        }

        for (unsigned int i = start; i < n; i++) {
            for (unsigned int j = 0; j < b2_cols; j++) {
                if (total_shift < 0) {
                        mpz_mul_2exp(dBs(i,j), dBs(i,j), -total_shift);
                } else {
                        mpz_div_2exp(dBs(i,j), dBs(i,j), total_shift);
                }

            }
        }
    }
    Matrix::copy(B.submatrix(0, end, 0, n), R.submatrix(0, end, 0, n));
    Matrix::copy(B.submatrix(end, n, end, n), B_next.submatrix(end, n, end, n));
    if (b2_cols > 0) {
        Matrix::copy(B2_sim.submatrix(0, end, 0, b2_cols), R.submatrix(0, end, n, n+b2_cols));
    }
    assert(B.is_upper_triangular());
    B_next = Matrix(ElementType::MPZ, B_next.nrows(), B_next.ncols());
}

void Heuristic2::update_R_representation() {
    auto sub_ind = sublattice_inds[0];
    unsigned int start = sub_ind.first;
    unsigned int end = sub_ind.second;

    // Update profile
    Profile sub_prof = L_subs[0].profile;
    for (unsigned int i = start; i < n; i++) {
        profile[i] = sub_prof[i-start];
    }

    double spread = profile.get_spread();
    unsigned int precision = get_precision_from_spread(spread);
    {
        // Assumes that U_subs, sublattice information is all up to date
        // Incorporates the sublattice reduction information into the
        // current state.
        assert(num_sublattices == 1);

        Matrix U_i = U_iters.back();
        U_i.set_identity();

        Matrix U_sub = U_subs[0];

        // Update U_i
        Matrix::copy(
            U_i.submatrix(start, end, start, end),
            U_sub
        );

        // Update B_next.
        // B_next will not necessarily be size reduced or triangular.
        MatrixMultiplication mm_b_next(
            B_next.submatrix(0, start, start, end),
            B.submatrix(0, start, start, end), U_sub, cc);
        mm_b_next.solve();
        Matrix::copy(B_next.submatrix(0, start, 0, start), B.submatrix(0, start, 0, start));
        Matrix::copy(
            B_next.submatrix(start, end, start, end),
            L_subs[0].basis()
        );

        // Do fused QR/ Size reduction operation
        Matrix U_sr(ElementType::MPZ, n, n);
        U_sr.set_identity();


        RelativeSizeReductionParams rsr_params(
            B_next.submatrix(0, start, 0, start),
            B_next.submatrix(0, start, start, end),
            U_sr.submatrix(0, start, start, end)
        );
        rsr_params.is_B1_upper_triangular = true;

        RelativeSizeReduction rsr(rsr_params, cc);
        rsr.solve();

        MatrixMultiplication mm_u_sr(U_i, U_i, U_sr, cc);
        mm_u_sr.solve();

        Matrix R(ElementType::MPFR, n-start, n-start+b2_cols, precision);

        Matrix::copy(
            R.submatrix(0, end-start, 0, end-start),
            B_next.submatrix(start, end, start, end)
        );
        if (b2_cols > 0) {
            Matrix::copy(
                R.submatrix(0, end-start, n-start, n-start+b2_cols),
                B2_sim.submatrix(start, end, 0, b2_cols)
            );
        }
        QRFactorization qr(R, cc);
        qr.solve();

        // Use R-factor to compute profile
        {
            MatrixData<mpfr_t> dR = R.data<mpfr_t>();
            for (unsigned int i = start; i < n; i++) {
                long int exp;
                double d = mpfr_get_d_2exp(&exp, dR(i-start, i-start), rnd);
                double newval = log2(fabs(d)) + exp;
                if (profile[i] != newval) {
                    lattice_changed = true;
                }
                profile[i] = newval;
            }
        }

        // Take R-factor, compress, and extract compressed basis to B.
        {
            // Assumes this->R is upper triangular and profile is valid
            // Writes compressed basis to B and updates profile.

            int *shifts = compression_iters.back();
            this->get_shifts_for_compression(shifts);
            int total_shift = shifts[0];
            for (unsigned int i = 0; i < n; i++) {
                shifts[i] = total_shift;
            }

            MatrixData<mpfr_t> dR = R.data<mpfr_t>();
            MatrixData<mpz_t> dB = B_next.data<mpz_t>();
            MatrixData<mpz_t> dBs;
            if (b2_cols > 0) {
                dBs = B2_sim.data<mpz_t>();
            }
            for (unsigned int j = 0; j < n + b2_cols; j++) {
                if (j < n) {
                    local_profile_offsets[j] += total_shift;
                    global_profile_offsets[j] += total_shift;
                    profile[j] -= total_shift;
                }

                for (unsigned int i = 0; i < n; i++) {
                    if (i >= start) {
                        if (j >= start) {
                            if (i > j) {
                                mpfr_set_zero(dR(i-start,j-start), 0);
                            } else {
                                mpfr_mul_2si(dR(i-start, j-start), dR(i-start, j-start), -total_shift, rnd);
                            }
                        }
                    } else {
                        if (j < n) {
                            if (i > j) {
                                mpz_set_ui(dB(i, j), 0);
                            } else {
                                if (total_shift < 0) {
                                    mpz_mul_2exp(dB(i,j), dB(i,j), -total_shift);
                                } else {
                                    mpz_div_2exp(dB(i,j), dB(i,j), total_shift);
                                }
                            }
                        } else {
                            if (total_shift < 0) {
                                mpz_mul_2exp(dBs(i,j-n), dBs(i,j-n), -total_shift);
                            } else {
                                mpz_div_2exp(dBs(i,j-n), dBs(i,j-n), total_shift);
                            }
                        }
                    }
                }
            }
        }
        Matrix::copy(B.submatrix(start, n, start, n), R.submatrix(0, n-start, 0, n-start));
        if (b2_cols > 0) {
            Matrix::copy(
                B2_sim.submatrix(start, n, 0, b2_cols),
                R.submatrix(0, n-start, n-start, n-start+b2_cols)
            );
        }
        Matrix::copy(B.submatrix(0, start, 0, n), B_next.submatrix(0, start, 0, n));

        assert(B.is_upper_triangular());
    }

    B_next = Matrix(ElementType::MPZ, B_next.nrows(), B_next.ncols());
}

void Heuristic2::update_all_representation() {
    // Assumes that U_subs, sublattice information is all up to date
    // Incorporates the sublattice reduction information into the
    // current state.
    assert(num_sublattices == 1);

    Matrix U_i = U_iters.back();
    U_i.set_identity();

    auto sub_ind = sublattice_inds[0];
    unsigned int start = sub_ind.first;
    unsigned int end = sub_ind.second;

    Matrix U_sub = U_subs[0];

    // Update U_i
    Matrix::copy(
        U_i.submatrix(start, end, start, end),
        U_sub
    );

    // Update U2 if necessary
    if (params.U2.ncols() > 0) {
        Matrix new_B2(ElementType::MPZ, n, b2_cols);

        // Accumulate U_L * U_R
        Matrix U_tmp(ElementType::MPZ, n, n);
        U_tmp.set_identity();

        for (unsigned int i = 0; i < U_iters.size() - 1; i++) {
            Matrix U_l_or_r = U_iters[i];
            MatrixMultiplication ui_mm(U_tmp, U_tmp, U_l_or_r, cc);
            ui_mm.solve();
        }

        MatrixMultiplication u2_mm (params.U2, U_tmp, U2s[0], true, cc);
        u2_mm.solve();
    }

    Profile sub_prof = L_subs[0].profile;
    for (unsigned int i = 0; i < n; i++) {
        profile[i] = sub_prof[i];
    }

    if (b2_cols > 0) {

        MatrixMultiplication mm_check(B2_orig, B_orig, this->params.U2, true, cc);
        mm_check.solve();

        Matrix::copy(params.B2, B2_orig);
        B2_orig = Matrix();
        B_orig = Matrix();
        B2_sim = Matrix();
    }
}

}
}