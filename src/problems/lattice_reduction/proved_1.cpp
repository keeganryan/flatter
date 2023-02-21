#include "proved_1.h"

#include "problems/lattice_reduction.h"
#include "problems/matrix_multiplication.h"
#include "problems/size_reduction.h"

#include <cassert>
#include <cstring>

namespace flatter {
namespace LatticeReductionImpl {

const std::string Proved1::impl_name() {return "Proved1";}

Proved1::Proved1(const LatticeReductionParams& p, const ComputationContext& cc) :
    Base(p, cc)
{
    _is_configured = false;
    configure(p, cc);
}

Proved1::~Proved1() {
    if (_is_configured) {
        unconfigure();
    }
}

void Proved1::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Proved1::configure(const LatticeReductionParams& p, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(p, cc);

    _is_configured = true;
}

void Proved1::solve() {
    log_start();

    if (m == n && M.is_upper_triangular()) {
        mon->profile_reset(n);

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
            profile[i] = sz;
        }

        unsigned int N = pow(2, ceil(log2(n)));
        Matrix M_padded(ElementType::MPZ, N, N);
        Matrix U_padded(ElementType::MPZ, N, N);
        Matrix::copy(M_padded.submatrix(0, n, 0, n), M);
        delete[] profile;
        profile = new double[N];

        mpz_t padval;
        mpz_init(padval);
        mpz_set_ui(padval, 1);
        mpz_mul_2exp(padval, padval, (unsigned int)maxv + n);

        MatrixData<mpz_t> dMp = M_padded.data<mpz_t>();
        for (unsigned int i = n; i < N; i++) {
            mpz_set(dMp(i,i), padval);
        }
        mpz_clear(padval);

        Lattice L_padded (M_padded);
        for (unsigned int i = 0; i < N; i++) {
            L_padded.profile[i] = profile[i];
        }

        LatticeReductionParams p2(L_padded, U_padded, rhf, true);

        if (rhf < 4/3. + 0.01) {
            rhf = 1.5;
        }
        double slope = log2(rhf) * 2;
        p2.goal = LatticeReductionGoal(N, slope, HERMITE_BEST_SLOPE);

        p2.proved = this->params.proved;

        p2.profile_offset = new double[N];
        for (unsigned int i = 0; i < N; i++) {
            p2.profile_offset[i] = 0;
        }

        LatticeReduction latred(p2, cc);
        latred.solve();

        Matrix::copy(M, M_padded.submatrix(0, n, 0, n));
        Matrix::copy(U, U_padded.submatrix(0, n, 0, n));
        for (unsigned int i = 0; i < n; i++) {
            params.L.profile[i] = p2.L.profile[i];
        }

        mm.solve();

        delete[] profile;
        delete[] p2.profile_offset;
    } else {
        assert(0);
    }

    log_end();
}

}
}