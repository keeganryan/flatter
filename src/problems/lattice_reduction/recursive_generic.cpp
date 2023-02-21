#include "recursive_generic.h"

#include "problems/fused_qr_size_reduction.h"
#include "problems/lattice_reduction.h"
#include "problems/matrix_multiplication.h"

#include <cassert>

namespace flatter {
namespace LatticeReductionImpl {

const std::string RecursiveGeneric::impl_name() {return "RecursiveGeneric";}

RecursiveGeneric::RecursiveGeneric(const LatticeReductionParams& p, const ComputationContext& cc) :
    Base(p, cc)
{
    _is_configured = false;
    configure(p, cc);
}

RecursiveGeneric::~RecursiveGeneric() {
    if (_is_configured) {
        unconfigure();
    }
}

void RecursiveGeneric::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void RecursiveGeneric::configure(const LatticeReductionParams& p, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(p, cc);

    assert(params.L.profile.is_valid());
    rnd = mpfr_get_default_rounding_mode();

    _is_configured = true;
}

void RecursiveGeneric::do_checks() {

}

unsigned int RecursiveGeneric::get_initial_precision() {
    unsigned int max_bits = 0;
    MatrixData<mpz_t> dM = M.data<mpz_t>();
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            unsigned int bits = mpz_sizeinbase(dM(i,j), 2);
            max_bits = std::max(bits, max_bits);
        }
    }
    return get_precision_from_spread(max_bits);
}

void RecursiveGeneric::init_compressed_B() {
    unsigned int precision = get_initial_precision();
    int* compression_init = new int[n];
    original_precision = precision;
    set_precision(precision);
    Matrix::copy(R, this->M);

    set_profile();
    log_profile();
    compression_iters.push_back(compression_init);
    compress_R();

    Matrix::copy(B, R);
}

unsigned int RecursiveGeneric::get_precision_from_spread(double spread) {
    unsigned int precision = 2.0 * spread + 10 + 30;//2 * n;
    return precision;
}

void RecursiveGeneric::init_solver() {
    // Ensure our representation is integer, upper triangular, size reduced,
    // and compressed

    do_checks();

    sublattice_inds.clear();

    B = Matrix(ElementType::MPZ, m, n);
    R = Matrix(ElementType::MPFR, m, n, 53);
    B_next = Matrix(ElementType::MPZ, m, n);
    profile = Profile(n);
    local_profile_offsets = new double[n];
    global_profile_offsets = new double[n];
    for (unsigned int i = 0; i<n; i++) {
        local_profile_offsets[i] = 0;
        global_profile_offsets[i] = profile_offset[i];
    }

    init_compressed_B();

    lattice_changed = true;
}

void RecursiveGeneric::final_sr() {
    {
        double max_prof = profile[0];
        // Check that we're close to size reduced
        MatrixData<mpz_t> dM = M.data<mpz_t>();
        // Specifically, the size of each vector should not be absurdly
        // large compared to the profile. (orthogonality defect small)

        for (unsigned int j = 0; j < n; j++) {
            unsigned int vec_sz = 0;
            for (unsigned int i = 0; i < m; i++) {
                unsigned int elem_sz = mpz_sizeinbase(dM(i,j), 2);
                vec_sz = std::max(vec_sz, elem_sz);
            }

            max_prof = std::max(max_prof, profile[j]);
        }
        for (unsigned int j = 0; j < params.B2.ncols(); j++) {
            MatrixData<mpz_t> dB = params.B2.data<mpz_t>();
            unsigned int vec_sz = 0;
            for (unsigned int i = 0; i < m; i++) {
                unsigned int elem_sz = mpz_sizeinbase(dB(i,j), 2);
                vec_sz = std::max(vec_sz, elem_sz);
            }
        }
    }

    Matrix U_tmp = Matrix(ElementType::MPZ, n, n);

    double spread = profile.get_spread();
    unsigned int new_precision = get_precision_from_spread(spread);
    set_precision(new_precision);
    FusedQRSizeReductionParams params(M, R, U_tmp);

    FusedQRSizeReduction fqrsr(params, cc);
    fqrsr.solve();
    
    MatrixMultiplication mm2(U, U, U_tmp, cc);
    mm2.solve();
}

void RecursiveGeneric::fini_solver() {
    collect_U();

    MatrixMultiplication mm(M, M, U, cc);
    mm.solve();

    // Update profile
    for (unsigned int i = 0; i < n; i++) {
        profile[i] += local_profile_offsets[i];
        global_profile_offsets[i] -= local_profile_offsets[i];
        local_profile_offsets[i] = 0;
    }

    // Do final size reduction
    final_sr();
    for (unsigned int i = 0; i < n; i++) {
        params.L.profile[i] = profile[i];
    }

    delete[] local_profile_offsets;
    delete[] global_profile_offsets;
}

void RecursiveGeneric::init_iter() {
    Matrix U_iter(ElementType::MPZ, n, n);
    U_iters.push_back(U_iter);
    int* compression_iter = new int[n];
    compression_iters.push_back(compression_iter);
    
    setup_sublattice_reductions();
}

void RecursiveGeneric::fini_iter() {
    cleanup_sublattice_reductions();
}

void RecursiveGeneric::log_profile() {
    double *profile_arr = new double[n];
    for (unsigned int i = 0; i < n; i++) {
        profile_arr[i] = profile[i];
    }
    mon->profile_update(profile_arr, global_profile_offsets, offset, offset + n);
    delete[] profile_arr;
}

bool RecursiveGeneric::is_reduced() {
    return params.goal.check(profile);
}

void RecursiveGeneric::setup_sublattice_reductions() {
    num_sublattices = 0;
}

void RecursiveGeneric::cleanup_sublattice_reductions() {
    sublattice_inds.clear();

    for (unsigned int i = 0; i < num_sublattices; i++) {
        if (!U_subs[i].is_identity()) {
            //lattice_changed = true;
        }
    }
    L_subs.clear();
    U_subs.clear();
    sub_params.clear();
}

void RecursiveGeneric::reduce_sublattices() {
    for (unsigned int i = 0; i < num_sublattices; i++) {
        LatticeReduction lr(sub_params[i], cc);
        lr.solve();
    }
}

void RecursiveGeneric::update_representation() {
    // Assumes that U_subs, sublattice information is all up to date
    // Incorporates the sublattice reduction information into the
    // current state.
    Matrix U_i = U_iters.back();
    U_i.set_identity();

    Matrix U_tmp(ElementType::MPZ, n, n);
    Matrix::copy(B_next, B);

    for (unsigned int i = 0; i < num_sublattices; i++) {
        auto sub_ind = sublattice_inds[i];
        unsigned int start = sub_ind.first;
        unsigned int end = sub_ind.second;

        Matrix U_sub = U_subs[i];

        // Update U_i
        U_tmp.set_identity();
        Matrix::copy(
            U_tmp.submatrix(start, end, start, end),
            U_sub
        );

        // Update B_next.
        // B_next will not necessarily be size reduced or triangular.
        MatrixMultiplication mm_b_next(B_next, B_next, U_tmp, cc);
        mm_b_next.solve();

        // Do fused QR/ Size reduction operation
        Matrix U_sr(ElementType::MPZ, n, n);

        // Size Reduce B_next, and compute its R-factor
        FusedQRSizeReductionParams params(B_next, R, U_sr);
        FusedQRSizeReduction fqrsr(params, cc);
        fqrsr.solve();

        MatrixMultiplication mm_u_sr(U_tmp, U_tmp, U_sr, cc);
        mm_u_sr.solve();

        MatrixMultiplication mm_u_i(U_i, U_i, U_tmp, cc);
        mm_u_i.solve();
    }

    // Use R-factor to compute profile
    set_profile();

    // Take R-factor, compress, and extract compressed basis to B.
    compress_R();
    Matrix::copy(B, R);
    assert(B.is_upper_triangular());
}

void RecursiveGeneric::collect_U() {
    assert(U_iters.size() == compression_iters.size() - 1);
    // Build U by multiplying all U_iters, starting from the end

    // We need to occasionally scale the right product so that the compression
    // leads to properly reducing the size of entries. In the Saruchi paper, this
    // is D^-1 U D

    U.set_identity();

    delete[] compression_iters.back();
    compression_iters.pop_back();

    while (U_iters.size() > 0) {
        Matrix U_iter = U_iters.back();

        int* D = compression_iters.back();
        MatrixData<mpz_t> dU = U_iter.data<mpz_t>();
        // Do D^-1 UD
        for (unsigned int i = 0; i < n; i++) {
            for (unsigned int j = 0; j < n; j++) {
                if (D[i] == D[j]) {
                    // Do nothing
                } else if (D[i] > D[j]) {
                    // then we should be below the diagonal and zero
                    assert(mpz_cmp_ui(dU(i,j), 0) == 0);
                } else {
                    // Then we should be above the diagonal and need to shift
                    mpz_mul_2exp(dU(i,j), dU(i,j), D[j] - D[i]);
                }
            }
        }


        MatrixMultiplication ui_mm (U, U_iter, U, cc);
        ui_mm.solve();
        U_iters.pop_back();

        delete[] compression_iters.back();
        compression_iters.pop_back();
    }

    assert(compression_iters.size() == 0);
}

void RecursiveGeneric::set_profile() {
    // Assumes this->R is up to date and upper triangular
    // Writes to profile array with current profile
    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    for (unsigned int i = 0; i < n; i++) {
        long int exp;
        double d = mpfr_get_d_2exp(&exp, dR(i, i), rnd);
        double newval = log2(fabs(d)) + exp;
        if (profile[i] != newval) {
            lattice_changed = true;
        }
        profile[i] = newval;
    }
}

unsigned int RecursiveGeneric::get_shifts_for_compression(int* shifts) {

    // Calculate how many bits are needed
    double *max_from_left = new double[n];
    double *min_from_right = new double[n];

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

    for (unsigned int j = 0; j < n; j++) {
        shifts[j] += new_shift;
    }

    delete[] max_from_left;
    delete[] min_from_right;

    return precision;
}

void RecursiveGeneric::compress_R() {
    // Assumes this->R is upper triangular and profile is valid
    // Writes compressed basis to B and updates profile.


    int *shifts = compression_iters.back();
    unsigned int precision = this->get_shifts_for_compression(shifts);

    MatrixData<mpfr_t> dR = R.data<mpfr_t>();
    for (unsigned int j = 0; j < n; j++) {
        local_profile_offsets[j] += shifts[j];
        global_profile_offsets[j] += shifts[j];
        profile[j] -= shifts[j];

        for (unsigned int i = 0; i < n; i++) {
            if (i > j) {
                mpfr_set_zero(dR(i,j), 0);
            } else {
                mpfr_mul_2si(dR(i, j), dR(i, j), -shifts[j], rnd);
            }
        }
    }

    set_precision(precision);
}

void RecursiveGeneric::set_precision(unsigned int prec) {
    this->precision = prec;
    R.set_precision(precision);
}

void RecursiveGeneric::solve() {
    log_start();

    /*
    Basic operations are:

    General initialization

    If not reduced:
    round_start
    Reduce sublattices
    Update internal representation.
    round end
    repeat

    General teardown

    When does precision change? Only during compression step? How do we make sure
    we don't lose precision during QR? It should be that precision never decreases.


    */
    init_solver();

    for (iterations = 0; ; iterations++) {
        if (is_reduced()) {
            break;
        }

        init_iter();

        reduce_sublattices();

        update_representation();

        fini_iter();
    }

    fini_solver();

    log_end();
}

}
}