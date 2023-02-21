#include "problems/lattice_reduction/fplll_impl.h"

#include <cassert>

namespace flatter {
namespace LatticeReductionImpl {

const std::string FPLLL::impl_name() {return "FPLLL";}

FPLLL::FPLLL(const LatticeReductionParams& p, const ComputationContext& cc) :
    Base(p, cc)
{
    _is_configured = false;
    rnd = mpfr_get_default_rounding_mode();
    configure(p, cc);
}

FPLLL::~FPLLL() {
    if (_is_configured) {
        unconfigure();
    }
}

void FPLLL::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void FPLLL::configure(const LatticeReductionParams& p, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(p, cc);

    assert(M.is_transposed() == false);
    assert(U.is_transposed() == false);
    assert(M.type() == ElementType::MPZ || M.type() == ElementType::INT64);
    assert(U.type() == ElementType::MPZ || U.type() == ElementType::INT64);

    _is_configured = true;
}

unsigned int FPLLL::get_block_size_for_rhf(double rhf) {
    // Numbers from Predicting Lattice Reduction (Gama)
    if (rhf > 1.025) {
        return 0;
    } else if (rhf > 1.015) {
        return 10;
    } else if (rhf > 1.0128) {
        return 20;
    } else if (rhf > 1.0109) {
        return 28;
    } else {
        return 35;
    }
}

void FPLLL::solve() {
    log_start();

    init_A();

    fplll::ZZ_mat<mpz_t> fplll_U;
    fplll_U.gen_identity(n);


    // Because of a bug in FPLLL, our code segfaults if parallelized enumeration
    // is concurrently performed on more than one lattice. The following line
    // disables parallelized enumeration.
    fplll::set_external_enumerator(nullptr);

    double rhf = this->params.goal.get_rhf();
    unsigned int bs = get_block_size_for_rhf(rhf);

    if (bs == 0) {

        // What delta do we require?
        // alpha ~= sqrt(2/sqrt(4*delta - 1))

        // By experimentation, delta ~= 0.18/sqrt(log2(rhf))
        double delta = 0.18 / sqrt(log2(this->rhf));
        delta = std::max(delta, 0.51);
        delta = std::min(delta, 0.999);

        int status = fplll::lll_reduction(
            this->A,
            fplll_U,
            delta,
            fplll::LLL_DEF_ETA,
            fplll::LLLMethod::LM_WRAPPER,
            fplll::FT_DEFAULT,
            0,
            fplll::LLL_DEFAULT
        );
        if (status != fplll::RED_SUCCESS) {
            printf("FPLLL failed unexpectedly. Reduction result %d\n", status);
            Matrix::print(M.transpose());
        }

        assert(status == fplll::RED_SUCCESS);
    } else {
        fplll::vector<fplll::Strategy> strategies;
        fplll::BKZParam param(bs, strategies);
        param.flags = fplll::BKZ_DEFAULT | fplll::BKZ_AUTO_ABORT;
        
        int status = fplll::bkz_reduction(
            &this->A, &fplll_U, param, fplll::FT_DEFAULT, 0
        );
        if (status != fplll::RED_SUCCESS) {
            assert(false);
        }
    }

    // Copy back to M.data
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++) {
            // FPLLL expects vectors-as-rows, so do transpose
            auto aji = A(j, i).get_data();
            if (M.type() == ElementType::MPZ) {
                mpz_set(M.data<mpz_t>()(i, j), aji);
            } else {
                M.data<int64_t>()(i, j) = mpz_get_si(aji);
            }
        }
    }
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            // FPLLL expects vectors-as-rows, so do transpose
            auto uji = fplll_U(j, i).get_data();
            if (U.type() == ElementType::MPZ) {
                mpz_set(U.data<mpz_t>()(i, j), uji);
            } else {
                U.data<int64_t>()(i, j) = mpz_get_si(uji);
            }
        }
    }

    if (true) {
        int gso_flags = fplll::GSO_OP_FORCE_LONG;
        fplll::ZZ_mat<mpz_t> u_inv;
        //fplll::MatGSO<fplll::Z_NR<mpz_t>, fplll::FP_NR<double>> m_gso(this->A, fplll_U, u_inv, gso_flags);
        fplll::MatGSO<fplll::Z_NR<mpz_t>, fplll::FP_NR<mpfr_t>> m_gso(this->A, fplll_U, u_inv, gso_flags);
        fplll::FP_NR<mpfr_t> tmp_r, tmp_mu;
        
        m_gso.update_gso();
            
        for (unsigned int i = 0; i < n; i++) {
            m_gso.get_r(tmp_r, i, i);
            m_gso.get_mu(tmp_mu, i, i);
            double rii2 = tmp_r.get_d();// / tmp_mu.get_d();
            double val = log2(fabs(rii2))/2.;
            params.L.profile[i] = val;
        }

        mon->profile_update(&params.L.profile[0], params.profile_offset, offset, offset + n);
    }

    log_end();
}


void FPLLL::init_A() {
    // Copy to LLL matrix
    A.resize(n, m);
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++) {
            // FPLLL expects vectors-as-rows, so do transpose
            auto aji = A(j, i).get_data();
            if (M.type() == ElementType::MPZ) {
                mpz_set(aji,  M.data<mpz_t>()(i, j));
            } else {
                mpz_set_si(aji, M.data<int64_t>()(i, j));
            }
        }
    }
}

}
}