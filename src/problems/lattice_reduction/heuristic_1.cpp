#include "heuristic_1.h"

#include "sublattice_split_2.h"

#include "problems/fused_qr_size_reduction.h"
//#include "lattice_reduction/lattice_reduction.h"
#include "problems/matrix_multiplication.h"
#include "problems/qr_factorization.h"
#include "problems/relative_size_reduction.h"

#include <cassert>

namespace flatter {
namespace LatticeReductionImpl {

const std::string Heuristic1::impl_name() {return "Heuristic1";}

Heuristic1::Heuristic1(const LatticeReductionParams& p, const ComputationContext& cc) :
    Heuristic2(p, cc)
{}

void Heuristic1::init_compressed_B() {
    assert(params.log_cond > 0);
    Matrix::copy(B, M);
    //profile[i] = M.prec();
    for (unsigned int i = 0; i < n; i++) {
        profile[i] = params.L.profile[i];
    }
    //profile[0] = M.prec();// + global_profile_offsets[0];

    int* compression_init = new int[n];
    for (unsigned int i = 0; i < n; i++) {
        compression_init[i] = 0;
    }
    compression_iters.push_back(compression_init);

    Matrix::copy(B2_sim, params.B2);

    log_profile();
}

void Heuristic1::setup_sublattice_reductions() {
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

    unsigned int bsub_rows = m - start;
    if (start == 0 && end == n) {
        bsub_rows = n;
    } else if (start == 0) {
        // Left reduction. Can we exclude certain rows?
        MatrixData<mpz_t> dB = B.data<mpz_t>();
        bool nonzero = false;
        while (bsub_rows > end && !nonzero) {
            unsigned int rbound;
            if (prec < 10000) {
                rbound = n;
            } else {
                rbound = end;
            }
            for (unsigned int j = 0; j < rbound; j++) {
                if (mpz_cmp_ui(dB(start + bsub_rows - 1, j), 0) != 0) {
                    nonzero = true;
                    break;
                }
            }
            if (!nonzero) {
                bsub_rows -= 1;
            }
        }
        if (bsub_rows < 3) {
            //bsub_rows = 3;
        }
    } else {
        // Right reduction
        MatrixData<mpz_t> dB = B.data<mpz_t>();
        bool nonzero = false;
        while (bsub_rows > n && !nonzero) {
            for (unsigned int j = 0; j < n; j++) {
                if (mpz_cmp_ui(dB(start + bsub_rows - 1, j), 0) != 0) {
                    nonzero = true;
                    break;
                }
            }
            if (!nonzero) {
                bsub_rows -= 1;
            }
        }
    }

    Matrix B_sub(ElementType::MPZ, bsub_rows, end-start);
    Lattice L_sub (B_sub);
    L_sub.profile = profile.subprofile(start, end);

    L_subs.push_back(L_sub);
    Matrix::copy(B_sub, B.submatrix(start, start+bsub_rows, start, end));

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
        B2 = Matrix(ElementType::MPZ, bsub_rows, b2_cols);
        U2 = Matrix(ElementType::MPZ, end-start, b2_cols);
        Matrix::copy(B2.submatrix(0, bsub_rows, 0, n-end), B.submatrix(0, bsub_rows, end, n));
        if (b2_cols > n-end) {
            Matrix::copy(
                B2.submatrix(0, bsub_rows, n-end, b2_cols),
                B2_sim.submatrix(0, bsub_rows, 0, b2_cols - (n-end))
            );
        }
        B2s.push_back(B2);
        U2s.push_back(U2);
    }

    if (start == 0 && end == n && params.B2.ncols() != 0) {
        // step all
        B2 = Matrix(ElementType::MPZ, n, b2_cols);
        U2 = Matrix(ElementType::MPZ, n, b2_cols);
        Matrix::copy(B2.submatrix(0, n, 0, b2_cols), B2_sim.submatrix(0, n, 0, b2_cols));
        B2s.push_back(B2);
        U2s.push_back(U2);
    }

    LatticeReductionParams params(L_sub, U_sub, rhf, true);

    LatticeReductionGoal goal1 = this->params.goal.subgoal(start, end);

    // Get mu_L and mu_R to approximate current_drop
    double mu_L = 0;
    double mu_R = 0;
    for (unsigned int i = 0; i < n/2; i++) {
        mu_L += profile[i] + global_profile_offsets[i];
    }
    mu_L /= (n / 2);

    double current_drop = mu_L - mu_R;
    if (current_drop < 0) {
        current_drop = profile.get_drop();
    }
    LatticeReductionGoal goal2 = LatticeReductionGoal::from_drop(
        end-start, current_drop / 6);

    if (!(start == 0 && end == n) && goal1.get_quality() < goal2.get_quality()) {
        params.goal = goal2;
    } else {
        params.goal = goal1;
    }

    params.split = this->params.split->get_child_split(0);
    assert(params.split != nullptr);
    params.proved = this->params.proved;
    params.offset = this->offset + start;
    params.profile_offset = &global_profile_offsets[start];
    params.phase = 1;
    params.log_cond = this->params.log_cond;
    params.aggressive_precision = this->params.aggressive_precision;

    params.B2 = B2;
    params.U2 = U2;

    if (start == 0 && end == n) {
        params.phase = 3;
    }

    sub_params.push_back(params);

    U_mul_inds.push_back(std::make_pair(start, end));
    assert(B.nrows() >= B.ncols());
}

bool Heuristic1::is_reduced() {
    if (this->params.split->stopping_point()) {
        return true;
    }
    return false;
}


void Heuristic1::update_L_representation() {
    assert(num_sublattices == 1);

    auto sub_ind = sublattice_inds[0];
    unsigned int start = sub_ind.first;
    unsigned int end = sub_ind.second;

    // Update profile
    Profile sub_prof = L_subs[0].profile;
    for (unsigned int i = 0; i < end; i++) {
        profile[i] = sub_prof[i];
    }

    unsigned int precision = get_precision_from_spread(params.log_cond);

    // Assumes that U_subs, sublattice information is all up to date
    // Incorporates the sublattice reduction information into the
    // current state.

    Matrix U_i = U_iters.back();
    Matrix U2 = U2s.back();
    Matrix B2 = B2s.back();

    U_i.set_identity();

    Matrix U_sub = U_subs[0];

    // Update U_i
    Matrix B_sub = L_subs.back().basis();
    unsigned int bsub_rows = B_sub.nrows();
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
            B2_sim.submatrix(0, bsub_rows, 0, b2_cols),
            B2.submatrix(0, bsub_rows, n-end, b2_cols + n-end)
        );
    }

    // Update B_next.
    Matrix::copy(B_next.submatrix(0, bsub_rows, 0, end), L_subs[0].basis());
    Matrix::copy(
        B_next.submatrix(0, bsub_rows, end, n),
        B2.submatrix(0, bsub_rows, 0, n-end)
    );
    if (bsub_rows < m) {
        Matrix::copy(
            B_next.submatrix(bsub_rows, m, 0, n),
            B.submatrix(bsub_rows, m, 0, n)
        );
    }

    // Do fused QR/ Size reduction operation

    // Size Reduce B_next, and compute its R-factor
    //FusedQRSizeReductionParams params(B_next, R, U_sr);
    //FusedQRSizeReduction fqrsr(params, cc);
    //fqrsr.solve();
    
    Matrix R(ElementType::MPFR, bsub_rows, n + b2_cols, precision);

    //set_precision(precision);
    Matrix::copy(R.submatrix(0, bsub_rows, 0, n), B_next.submatrix(0, bsub_rows, 0, n));
    if (b2_cols > 0) {
        Matrix::copy(R.submatrix(0, bsub_rows, n, n+b2_cols), B2_sim.submatrix(0, bsub_rows, 0, b2_cols));
    }
    
    QRFactorization qr(R, cc);
    qr.solve();

    // Use R-factor to compute profile
    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    for (unsigned int i = 0; i < std::min(n, bsub_rows); i++) {
        long int exp;
        double d = mpfr_get_d_2exp(&exp, dR(i, i), rnd);
        double newval = log2(fabs(d)) + exp;

        assert(!std::isnan(newval));
        profile[i] = newval;
    }
    if (n == 3 && end == 2 && bsub_rows >= 3) {
        long int exp;
        double d = mpfr_get_d_2exp(&exp, dR(2, 2), rnd);
        double newval = log2(fabs(d)) + exp;

        assert(!std::isnan(newval));
        profile[2] = newval;
    }

    // Take R-factor, compress, and extract compressed basis to B.
    //compress_R();
    {
        int *shifts = compression_iters.back();
        this->get_shifts_for_compression(shifts);
        int total_shift = shifts[0];
        for (unsigned int i = 0; i < n; i++) {
            shifts[i] = total_shift;
        }
        //total_shift = -precision;

        MatrixData<mpfr_t> dR = R.data<mpfr_t>();
        //MatrixData<mpz_t> dB = B_next.data<mpz_t>();
        MatrixData<mpz_t> dBs;
        if (b2_cols > 0) {
            dBs = B2_sim.data<mpz_t>();
        }
        for (unsigned int i = 0; i < n; i++) {
            local_profile_offsets[i] += total_shift;
            global_profile_offsets[i] += total_shift;
            profile[i] -= total_shift;
        }

        for (unsigned int i = 0; i < bsub_rows; i++) {
            for (unsigned int j = 0; j < n + b2_cols; j++) {
                if (j < i) {
                    mpfr_set_zero(dR(i,j), 0);
                } else {
                    mpfr_mul_2si(dR(i,j), dR(i,j), -total_shift, rnd);
                }
            }
        }

        MatrixData<mpz_t> dB = B_next.data<mpz_t>();
        for (unsigned int i = bsub_rows; i < m; i++) {
            for (unsigned int j = 0; j < n; j++) {
                if (total_shift < 0) {
                    mpz_mul_2exp(dB(i,j), dB(i,j), -total_shift);
                } else {
                    mpz_div_2exp(dB(i,j), dB(i,j), total_shift);
                }
            }
        }

        for (unsigned int i = bsub_rows; i < m; i++) {
            for (unsigned int j = 0; j < b2_cols; j++) {
                if (total_shift < 0) {
                        mpz_mul_2exp(dBs(i,j), dBs(i,j), -total_shift);
                } else {
                        mpz_div_2exp(dBs(i,j), dBs(i,j), total_shift);
                }

            }
        }

        Matrix::copy(B_next.submatrix(0, bsub_rows, 0, n), R.submatrix(0, bsub_rows, 0, n));
    }
    Matrix::copy(B, B_next);
    if (b2_cols > 0) {
        Matrix::copy(B2_sim.submatrix(0, bsub_rows, 0, b2_cols), R.submatrix(0, bsub_rows, n, n+b2_cols));
    }
}

void Heuristic1::update_R_representation() {
    auto sub_ind = sublattice_inds[0];
    unsigned int start = sub_ind.first;
    unsigned int end = sub_ind.second;

    // Update profile
    Profile sub_prof = L_subs[0].profile;
    for (unsigned int i = start; i < n; i++) {
        profile[i] = sub_prof[i-start];
    }

    unsigned int precision = get_precision_from_spread(params.log_cond);

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

        Matrix B_sub = L_subs.back().basis();
        unsigned int bsub_rows = B_sub.nrows();
        // Update B_next.
        // B_next will not necessarily be size reduced or triangular.
        MatrixMultiplication mm_b_next(
            B_next.submatrix(0, start, start, end),
            B.submatrix(0, start, start, end), U_sub, cc);
        mm_b_next.solve();
        Matrix::copy(B_next.submatrix(0, start, 0, start), B.submatrix(0, start, 0, start));
        Matrix::copy(B_next.submatrix(start, start+bsub_rows, start, end), L_subs[0].basis());

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

        Matrix R(ElementType::MPFR, bsub_rows, n-start+b2_cols, precision);

        Matrix::copy(
            R.submatrix(0, bsub_rows, 0, end-start),
            B_next.submatrix(start, start + bsub_rows, start, end)
        );
        if (b2_cols > 0) {
            Matrix::copy(
                R.submatrix(0, bsub_rows, n-start, n-start+b2_cols),
                B2_sim.submatrix(start, start+bsub_rows, 0, b2_cols)
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
                assert(!std::isnan(newval));
                profile[i] = newval;
            }
        }

        // Take R-factor, compress, and extract compressed basis to B.
        //compress_R();
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
            }

            for (unsigned int i = 0; i < start; i++) {
                for (unsigned int j = 0; j < n; j++) {
                    if (i > j) {
                        mpz_set_ui(dB(i, j), 0);
                    } else {
                        if (total_shift < 0) {
                            mpz_mul_2exp(dB(i,j), dB(i,j), -total_shift);
                        } else {
                            mpz_div_2exp(dB(i,j), dB(i,j), total_shift);
                        }
                    }
                }
            }
            for (unsigned int i = 0; i < bsub_rows; i++) {
                for (unsigned int j = 0; j < n; j++) {
                    if (j >= start) {
                        if (i+start > j) {
                            mpfr_set_zero(dR(i,j-start), 0);
                        } else {
                            mpfr_mul_2si(dR(i, j-start), dR(i, j-start), -total_shift, rnd);
                        }
                    }
                }
            }
            for (unsigned int j = 0; j < b2_cols; j++) {
                for (unsigned int i = 0; i < m; i++) {
                    if (i >= start && i < start + bsub_rows) {
                        mpfr_mul_2si(dR(i-start, n-start+j), dR(i-start, n-start+j), -total_shift, rnd);
                    } else {
                        if (total_shift < 0) {
                            mpz_mul_2exp(dBs(i,j), dBs(i,j), -total_shift);
                        } else {
                            mpz_div_2exp(dBs(i,j), dBs(i,j), total_shift);
                        }
                    }
                }
            }
        }
        Matrix::copy(B.submatrix(start, start+bsub_rows, start, n), R.submatrix(0, bsub_rows, 0, n-start));
        if (b2_cols > 0) {
            Matrix::copy(
                B2_sim.submatrix(start, start+bsub_rows, 0, b2_cols),
                R.submatrix(0, bsub_rows, n-start, n-start+b2_cols)
            );
        }
        Matrix::copy(B.submatrix(0, start, 0, n), B_next.submatrix(0, start, 0, n));
    }
}

void Heuristic1::update_all_representation() {
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
        Matrix new_B2(ElementType::MPZ, m, b2_cols);

        // Accumulate U_L * U_R
        Matrix U_tmp = Matrix(ElementType::MPZ, n, n);
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
        Matrix B2p(ElementType::MPZ, m, b2_cols);
        Matrix::copy(B2p, B2_orig);

        MatrixMultiplication mm_check(B2p, B_orig, this->params.U2, true, cc);
        mm_check.solve();
        Matrix::copy(params.B2, B2p);
    }
}

}
}