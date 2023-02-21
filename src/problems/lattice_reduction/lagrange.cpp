#include "lagrange.h"

#include <cassert>

namespace flatter {
namespace LatticeReductionImpl {

const std::string Lagrange::impl_name() {return "Lagrange";}

Lagrange::Lagrange(const LatticeReductionParams& p, const ComputationContext& cc) :
    Base(p, cc)
{
    _is_configured = false;
    rnd = mpfr_get_default_rounding_mode();
    configure(p, cc);
}

Lagrange::~Lagrange() {
    if (_is_configured) {
        unconfigure();
    }
}

void Lagrange::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Lagrange::configure(const LatticeReductionParams& p, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(p, cc);
    assert(n <= 2 && m == n);

    _is_configured = true;
}

void Lagrange::solve() {
    log_start();
    if (n == 1) {
        U.set_identity();
        log_end();
        return;
    }
    WorkspaceBuffer<mpfr_t> wsb(8, prec);
    mpz_t q_z, tmp_z;
    mpz_init(q_z);
    mpz_init(tmp_z);


    mpfr_t* local_ws = wsb.walloc(7);

    mpfr_t& a_len = local_ws[0];
    mpfr_t& b_len = local_ws[1];
    mpfr_t& q = local_ws[2];
    mpfr_t& tmp = local_ws[3];
    mpfr_t& adotb = local_ws[4];
    mpfr_t& det = local_ws[5];
    mpfr_t& limit = local_ws[6];

    Matrix ZM(ElementType::MPZ, M.nrows(), M.ncols());
    Matrix ZU(ElementType::MPZ, U.nrows(), U.ncols());
    Matrix::copy(ZM, M);
    Matrix::copy(ZU, U);
    ZU.set_identity();

    MatrixData<mpz_t> dM = ZM.data<mpz_t>();
    MatrixData<mpz_t> dU = ZU.data<mpz_t>();
    
    mpz_t& a0 = dM(0,0);
    mpz_t& a1 = dM(1,0);
    mpz_t& b0 = dM(0,1);
    mpz_t& b1 = dM(1,1);

    mpz_t& U00 = dU(0,0);
    mpz_t& U01 = dU(0,1);
    mpz_t& U10 = dU(1,0);
    mpz_t& U11 = dU(1,1);

    // Given the rhf, what is the maximum vector length for
    // the first vector?
    double hermite_factor = pow(this->rhf, 4);
    // Compute determinant
    mpfr_set_z(local_ws[0], a0, rnd);
    mpfr_mul_z(local_ws[0], local_ws[0], b1, rnd);
    mpfr_set_z(local_ws[1], b0, rnd);
    mpfr_mul_z(local_ws[1], local_ws[1], a1, rnd);
    mpfr_sub(local_ws[0], local_ws[0], local_ws[1], rnd);
    mpfr_abs(det, local_ws[0], rnd);
    // ||b1||^2 < rhf^2^2 * sqrt(det)^2
    mpfr_set(limit, det, rnd);
    mpfr_mul_d(limit, limit, hermite_factor, rnd);

    // Make sure both columns of M are nonzero
    norm2(a_len, a0, a1, wsb);
    assert(!mpfr_zero_p(a_len));
    norm2(b_len, b0, b1, wsb);
    assert(!mpfr_zero_p(b_len));

    if (mpfr_cmp(a_len, b_len) < 0) {
        // Perform first swap
        mpz_swap(a0, b0);
        mpz_swap(a1, b1);
        mpfr_swap(a_len, b_len);
        mpz_swap(U00, U01);
        mpz_swap(U10, U11);
    }

    while (1) {
        // Compute <a, b>
        mpfr_set_z(adotb, a0, rnd);
        mpfr_mul_z(adotb, adotb, b0, rnd);
        mpfr_set_z(tmp, a1, rnd);
        mpfr_mul_z(tmp, tmp, b1, rnd);
        mpfr_add(adotb, adotb, tmp, rnd);

        mpfr_div(q, adotb, b_len, rnd);
        mpfr_round(q, q);
        mpfr_get_z(q_z, q, rnd);

        // Perform swap
        mpz_swap(a0, b0);
        mpz_swap(a1, b1);
        mpfr_swap(a_len, b_len);

        // vector A now has value B
        // vector B now has value A, but should have A - B * q
        mpz_mul(tmp_z, a0, q_z);
        mpz_sub(b0, b0, tmp_z);
        mpz_mul(tmp_z, a1, q_z);
        mpz_sub(b1, b1, tmp_z);

        // new a_len is old b_len, so doesn't need to be updated
        // Update new b_len
        norm2(b_len, b0, b1, wsb);

        // Need to update U, which is equivalent to a multiplication by
        // [e f] [0  1]   [f e-qf]
        // [g h] [1 -q] = [h g-qh]
        mpz_swap(U00, U01);
        mpz_swap(U10, U11);
        mpz_mul(tmp_z, U00, q_z);
        mpz_sub(U01, U01, tmp_z);
        mpz_mul(tmp_z, U10, q_z);
        mpz_sub(U11, U11, tmp_z);

        if (mpfr_cmp(a_len, b_len) <= 0) {
            break;
        }
    }

    // a_len is ||a||^2
    // b_len is ||b||^2
    double a, d;
    long a_exp, d_exp;
    a = mpfr_get_d_2exp(&a_exp, a_len, rnd);
    d = mpfr_get_d_2exp(&d_exp, det, rnd);

    params.L.profile[0] = (a_exp + log2(a)) / 2;
    params.L.profile[1] = d_exp + log2(d) - (a_exp - log2(a)) / 2;
    mon->profile_update(
        &params.L.profile[0],
        profile_offset,
        offset,
        offset + 2);

    {
        // Check that we are fully size reduced
        mpz_t dot;
        mpz_init(dot);

        mpz_mul(dot, a0, b0);
        mpz_mul(tmp_z, a1, b1);
        mpz_add(dot, dot, tmp_z);
        // dot = <a, b>
        // Want <a, b> / <a,a> < 1
        // Equivalently <a,b> < <a,a>
        mpz_mul(tmp_z, a0, a0);
        mpz_mul(q_z, a1, a1);
        mpz_add(tmp_z, tmp_z, q_z);

        mpz_tdiv_q(q_z, dot, tmp_z);
        assert(mpz_cmpabs(dot, tmp_z) <= 0);

        mpz_clear(dot);
    }


    wsb.wfree(local_ws, 7);
    mpz_clear(tmp_z);
    mpz_clear(q_z);

    Matrix::copy(M, ZM);
    Matrix::copy(U, ZU);

    log_end();
}

void Lagrange::norm2(mpfr_t& r, mpz_t& x1, mpz_t& x2, WorkspaceBuffer<mpfr_t>& ws) {
    mpfr_t* local_ws = ws.walloc(1);
    mpfr_set_z(r, x1, rnd);
    mpfr_sqr(r, r, rnd);
    mpfr_set_z(local_ws[0], x2, rnd);
    mpfr_sqr(local_ws[0], local_ws[0], rnd);
    mpfr_add(r, r, local_ws[0], rnd);
    ws.wfree(local_ws, 1);
}

}
}