#include "threaded_3.h"

#include <cassert>

#include "problems/lattice_reduction.h"

namespace flatter {
namespace LatticeReductionImpl {

const std::string Threaded3::impl_name() {return "Threaded3";}

Threaded3::Threaded3(const LatticeReductionParams& p, const ComputationContext& cc) :
    Heuristic3(p, cc)
{}

void Threaded3::init_iter() {
    Heuristic3::init_iter();
    setup_taskvars();
}

void Threaded3::fini_iter() {
    cleanup_taskvars();
    Heuristic3::fini_iter();
}

void Threaded3::lr(unsigned int s_ind) {
    Tile t = tiles[tiles_to_reduce[s_ind]];
    unsigned int rank = t.end - t.start;
    unsigned int prec = L_subs[s_ind].basis().prec();
    ComputationContext cc = this->cc;
    if (prec < 1000 && rank < 100) {
        cc = ComputationContext(1);
    }
    LatticeReduction lr(sub_params[s_ind], cc);
    lr.solve();
}

void Threaded3::reduce_and_update() {
    char* taskvars;
    taskvars = this->taskvars;

    // Assumes that U_subs, sublattice information is all up to date
    // Incorporates the sublattice reduction information into the
    // current state.
    Matrix U_i = U_iters.back();
    U_i.set_identity();
    U_sr.set_identity();

    for (unsigned int i = 0; i < num_sublattices; i++) {
        #pragma omp task \
            firstprivate(i) \
            depend(in: taskvars[tvi('l', 0, 0)]) \
            depend(out: taskvars[tvi('u',i,i)]) \
            if (use_tasks)
        {
            lr(i);
            unsigned int ind = tiles_to_reduce[i];
            Matrix::copy(
                get_tile(U_i, ind, ind),
                U_subs[i]
            );
        }
    }
    #pragma omp taskwait

    // Update B_next.
    // B_next will not necessarily be size reduced or triangular.

    for (unsigned int i = 0; i < num_tiles; i++) {
        for (unsigned int j = 0; j <= i; j++) {
            // Requires U[i,i]
            // Updates B[j,i], B2[j, i]
            #pragma omp task \
                depend(in: taskvars[tvi('u', i, i)]) \
                depend(out: taskvars[tvi('b',j,i)]) \
                depend(out: taskvars[tvi('n',j,i)]) \
                if (use_tasks)
            update_b(j, i);
        }
    }

    // Requires completion of all lr(i)
    // This is accomplished by having all lr(i) depend on one value as input,
    // then claim this task will write to that value as output.
    // Used to update precision, but not right now

    #pragma omp task \
        depend(out: taskvars[tvi('l',0,0)]) \
        depend(out: taskvars[tvi('p',0,0)]) \
        if (use_tasks)
    {}

    // QR factorize
    for (unsigned int i = 0; i < num_tiles; i++) {
            // Requires Prec, U[i,i] (lr completion)
            // Updates R[i,i], B2[i,i]
            #pragma omp task \
                depend(in: taskvars[tvi('p',0,0)]) \
                depend(in: taskvars[tvi('u',i,i)]) \
                depend(out: taskvars[tvi('r', i, i)]) \
                depend(out: taskvars[tvi('n', i, i)]) \
                if (use_tasks)
            qr(i);
    }

    for (unsigned int i = 0; i < num_tiles; i++) {
        unsigned int tile_col = num_tiles - 1 - i;

        for (unsigned int j = 0; j < tile_col; j++) {
            // Work from bottom up
            unsigned int tile_row = tile_col - j - 1;

            // Requires B2[tile_row, tile_col], R[tile_row, tile_row]
            // Updates B2[tile_row, tile_col], U[tile_row, tile_col]
            #pragma omp task \
                depend(inout: taskvars[tvi('n',tile_row, tile_col)]) \
                depend(in: taskvars[tvi('r', tile_row, tile_row)]) \
                depend(out: taskvars[tvi('u', tile_row, tile_col)]) \
                if (use_tasks)
            sr(U_sr, tile_row, tile_col);

            for (unsigned int k = 0; k < tile_row; k++) {
                // Update B and U_sr
                // Requires B[k, tile_row], U[tile_row, tile_col], B2[k, tile_col]
                // Updates B2[k, tile_col]
                #pragma omp task depend(in: taskvars[tvi('b', k, tile_row)]) \
                                depend(in: taskvars[tvi('u', tile_row, tile_col)]) \
                                depend(inout: taskvars[tvi('n', k, tile_col)]) \
                                if (use_tasks)
                update_b_next(U_sr, k, tile_row, tile_col);
            }

            // Requires U[tile_row, tile_col], U[tile_row, tile_row]
            // Updates U[tile_row, tile_col]
            #pragma omp task \
                depend(inout: taskvars[tvi('u', tile_row, tile_col)]) \
                depend(in: taskvars[tvi('u', tile_row, tile_row)]) \
                if (use_tasks)
            update_u(U_sr, tile_row, tile_row, tile_col);
        }
    }

    if (use_tasks) {
        #pragma omp taskwait
    }

    // Use R-factor to compute profile
    set_profile();
    // Take R-factor, compress, and extract compressed basis to B.
    compress_R();
    Matrix::copy(B, R);

    //check_R();

    assert(B.is_upper_triangular());
}

void Threaded3::setup_taskvars() {
    // How many taskvars do we need for synchronization?
    //   B(i,j)                 k^2 (not k(k+1)/2 for ease of coding)
    //   B_next(i,j)            k^2 (not k(k+1)/2)
    //   U(i,j)                 k^2 (not k(k+1)/2)
    //   R(i,i)                 k
    //   PROFILE                1
    //   PREC                   1
    unsigned int k = num_tiles;
    //unsigned int num_needed = 3 * (k*(k+1)/2) + k + 1 + 1;
    unsigned int num_needed = 3*k*k + k + 1 + 1;
    taskvars = new char[num_needed];
}

void Threaded3::cleanup_taskvars() {
    delete[] taskvars;
}

unsigned int Threaded3::tvi(char c, unsigned int i, unsigned int j) {
    // Given a brief description of a task input/output, return an
    // index into the taskvar array
    unsigned int k = num_tiles;
    unsigned int offset = 0;
    switch (c)
    {
    case 'b':
        offset = 0;
        offset += i*k + j;
        break;
    case 'n':
        offset = k*k;
        offset += i*k + j;
        break;
    case 'u':
        offset = 2*k*k;
        offset += i*k + j;
        break;
    case 'r':
        assert(i == j);
        offset = 3*k*k;
        offset += i;
        break;
    case 'l':
        offset = 3*k*k+k;
        break;
    case 'p':
        offset = 3*k*k+k+1;
        break;
    default:
        assert(0);
    }
    return offset;
}

void Threaded3::main_loop() {
    for (iterations = 0; ; iterations++) {
        if (is_reduced()) {
            break;
        }

        init_iter();

        reduce_and_update();

        fini_iter();
    }
}

void Threaded3::solve() {
    log_start();

    init_solver();

    use_tasks = true;

    main_loop();

    fini_solver();

    log_end();
}

}
}