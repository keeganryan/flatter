#include "schoenhage.h"

#include <cassert>

#include "problems/matrix_multiplication.h"

namespace flatter {
namespace LatticeReductionImpl {

const std::string Schoenhage::impl_name() {return "Schoenhage";}

Schoenhage::Schoenhage(const LatticeReductionParams& p, const ComputationContext& cc) :
    Base(p, cc)
{
    _is_configured = false;
    configure(p, cc);
}

Schoenhage::~Schoenhage() {
    if (_is_configured) {
        unconfigure();
    }
}

void Schoenhage::unconfigure() {
    assert(_is_configured);

    _is_configured = false;
}

void Schoenhage::configure(const LatticeReductionParams& p, const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    Base::configure(p, cc);
    assert(n <= 2);
    assert(M.type() == ElementType::MPZ);
    assert(U.type() == ElementType::MPZ);

    _is_configured = true;
}

void Schoenhage::xgcd(mpz_t& a, mpz_t& b, mpz_t& g, mpz_t& x, mpz_t& y) {
    // The extended Euclidean algorithm shouldn't live in the Schoenhage implementation,
    // but it would require substantial rearchitecting and testing to relocate it

    // Given a, b as input, generate g, x, y such that
    //   g = GCD(a, b)
    //   g = x*a + b*y
    mpz_t old_r, r, old_s, s, old_t, t, q, tmp;
    mpz_init(old_r);
    mpz_init(r);
    mpz_init(old_s);
    mpz_init(s);
    mpz_init(old_t);
    mpz_init(t);
    mpz_init(q);
    mpz_init(tmp);

    mpz_set(old_r, a);
    mpz_set(r, b);
    mpz_set_ui(old_s, 1);
    mpz_set_ui(s, 0);
    mpz_set_ui(old_t, 0);
    mpz_set_ui(t, 1);

    // We have the invariants
    //     old_r == old_s * a + old_t * b
    //     r == s * a + t * b
    while (mpz_sgn(r) != 0) {
        mpz_fdiv_q(q, old_r, r);

        // Need old_r = r and r = old_r - q * r
        mpz_mul(tmp, q, r);
        mpz_sub(old_r, old_r, tmp);
        mpz_swap(r, old_r);

        // Need old_s = s and s = old_s - q * s
        mpz_mul(tmp, q, s);
        mpz_sub(old_s, old_s, tmp);
        mpz_swap(s, old_s);

        // Need old_t = t and t = old_t - q * t
        mpz_mul(tmp, q, t);
        mpz_sub(old_t, old_t, tmp);
        mpz_swap(t, old_t);
    }

    mpz_set(g, old_r);
    mpz_set(x, old_s);
    mpz_set(y, old_t);

    mpz_clear(old_r);
    mpz_clear(r);
    mpz_clear(old_s);
    mpz_clear(s);
    mpz_clear(old_t);
    mpz_clear(t);
    mpz_clear(q);
    mpz_clear(tmp);
}

void Schoenhage::simple_step(mpz_t& a, mpz_t& b, mpz_t& c, unsigned int m, mpz_t& t, bool& is_low_step) {
    // Do a single step above 2**m.
    // According to the paper, t >= 0, but this seems like it is not always the case.
    // If b < 2 * s, then we need to add to b such that B = b - 2 * a * t >= 2 * s
    mpz_t d, tmp1, tmp2;
    mpz_init(d);
    mpz_init(tmp1);
    mpz_init(tmp2);

    // Compute discriminant d = b**2 - 4*a*c
    mpz_mul(d, b, b);
    mpz_mul(tmp1, a, c);
    mpz_mul_2exp(tmp1, tmp1, 2);
    mpz_sub(d, d, tmp1);

    if (mpz_cmp(a, c) < 0) {
        // Low step
        is_low_step = true;
        // tmp1 is d + 4as
        mpz_mul_2exp(tmp1, a, m + 2);
        mpz_add(tmp1, tmp1, d);
        // tmp2 is 4s^2
        mpz_set_ui(tmp2, 1);
        mpz_mul_2exp(tmp2, tmp2, 2*m+2);
        if (mpz_cmp(tmp1, tmp2) < 0) {
            // Set r to 2*s. Use tmp1 for r
            mpz_set_ui(tmp1, 1);
            mpz_mul_2exp(tmp1, tmp1, m+1);
        } else {
            // Set r to ceil(sqrt(d + 4*a*s))
            mpz_sub_ui(tmp1, tmp1, 1);
            mpz_sqrt(tmp1, tmp1);
            mpz_add_ui(tmp1, tmp1, 1);
        }
        // Set t to (b - r) // (2 * a)
        mpz_sub(t, b, tmp1);
        mpz_fdiv_q(t, t, a);
        mpz_fdiv_q_2exp(t, t, 1);

        // Update a, b, c
        // a = a
        // b = b - 2 * a * t
        // c = c - b * t + a * t**2
        mpz_mul(tmp1, a, t); // tmp1 = a*t
        mpz_mul(tmp2, tmp1, t); // tmp2 = a*t*t
        mpz_mul_2exp(tmp1, tmp1, 1); // tmp1 = 2*a*t
        mpz_add(c, c, tmp2); // c = c + a*t**2
        mpz_sub(tmp2, b, tmp1); // tmp2 = b - 2*a*t
        mpz_mul(tmp1, b, t);
        mpz_sub(c, c, tmp1); // c = c - b*t + a*t**2
        mpz_set(b, tmp2); // b = b - 2*a*t
        
        // Flip sign of t to indicate this is a low step
        mpz_neg(t, t);
    } else {
        // High step
        // tmp1 is d + 4cs
        is_low_step = false;
        mpz_mul_2exp(tmp1, c, m + 2);
        mpz_add(tmp1, tmp1, d);
        // tmp2 is 4s^2
        mpz_set_ui(tmp2, 1);
        mpz_mul_2exp(tmp2, tmp2, 2*m+2);
        if (mpz_cmp(tmp1, tmp2) < 0) {
            // Set r to 2*s. Use tmp1 for r
            mpz_set_ui(tmp1, 1);
            mpz_mul_2exp(tmp1, tmp1, m+1);
        } else {
            // Set r to ceil(sqrt(d + 4*c*s))
            mpz_sub_ui(tmp1, tmp1, 1);
            mpz_sqrt(tmp1, tmp1);
            mpz_add_ui(tmp1, tmp1, 1);
        }
        // Set t to (b - r) // (2 * c)
        mpz_sub(t, b, tmp1);
        mpz_fdiv_q(t, t, c);
        mpz_fdiv_q_2exp(t, t, 1);

        // Update a, b, c
        // a = a - b * t + c * t**2
        // b = b - 2 * c * t
        // c = c
        mpz_mul(tmp1, c, t); // tmp1 = c*t
        mpz_mul(tmp2, tmp1, t); // tmp2 = c*t*t
        mpz_mul_2exp(tmp1, tmp1, 1); // tmp1 = 2*c*t
        mpz_add(a, a, tmp2); // a = a + c*t**2
        mpz_sub(tmp2, b, tmp1); // tmp2 = b - 2*c*t
        mpz_mul(tmp1, b, t);
        mpz_sub(a, a, tmp1); // a = a - b*t + c*t**2
        mpz_set(b, tmp2);
    }

    mpz_clear(d);
    mpz_clear(tmp1);
    mpz_clear(tmp2);
}

bool Schoenhage::is_minimal(mpz_t& a, mpz_t& b, mpz_t& c, unsigned int m) {
    if ((mpz_sgn(a) <= 0 || mpz_sizeinbase(a, 2) <= m) ||
        (mpz_sgn(b) <= 0 || mpz_sizeinbase(b, 2) <= m) ||
        (mpz_sgn(c) <= 0 || mpz_sizeinbase(c, 2) <= m)) {
        return false;
    }
    mpz_t tmp;
    mpz_init(tmp);

    mpz_sub(tmp, a, b);
    mpz_add(tmp, tmp, c);
    if (mpz_sgn(tmp) <= 0 || mpz_sizeinbase(tmp, 2) <= m) {
        mpz_clear(tmp);
        return true;
    }

    // Check if b - 2a >= 2*s
    mpz_mul_2exp(tmp, a, 1);
    mpz_sub(tmp, b, tmp);
    if (mpz_sgn(tmp) > 0 && mpz_sizeinbase(tmp, 2) > m + 1) {
        mpz_clear(tmp);
        return false;
    }

    // Check if b - 2c >= 2*s
    mpz_mul_2exp(tmp, c, 1);
    mpz_sub(tmp, b, tmp);
    if (mpz_sgn(tmp) > 0 && mpz_sizeinbase(tmp, 2) > m + 1) {
        mpz_clear(tmp);
        return false;
    }

    mpz_clear(tmp);
    return true;
}

void Schoenhage::nonrecursive(mpz_t& a, mpz_t& b, mpz_t& c, unsigned int m, Matrix U) {
    // Update a, b, c minimal above 2**m and return Matrix U transforming
    // input to output
    U.set_identity();
    MatrixData<mpz_t> dU = U.data<mpz_t>();
    mpz_t& U00 = dU(0,0);
    mpz_t& U01 = dU(0,1);
    mpz_t& U10 = dU(1,0);
    mpz_t& U11 = dU(1,1);

    mpz_t t, tmp;
    mpz_init(t);
    mpz_init(tmp);

    // Do simple steps until t = 0
    bool is_low_step;
    while(true) {
        simple_step(a, b, c, m, t, is_low_step);
        if (mpz_cmp_si(t, 0) == 0) {
            // t == 0
            break;
        } else if (is_low_step) {
            // low step
            // We must right multiply U by
            // [1 -(-t)]
            // [0  1]
            // That is, add t copies of U0 to U1
            mpz_mul(tmp, U00, t);
            mpz_add(U01, U01, tmp);
            mpz_mul(tmp, U10, t);
            mpz_add(U11, U11, tmp);
        } else {
            // high step
            // We must right multiply U by
            // [1  0]
            // [-t 1]
            mpz_mul(tmp, U01, t);
            mpz_sub(U00, U00, tmp);
            mpz_mul(tmp, U11, t);
            mpz_sub(U10, U10, tmp);
        }
    }

    mpz_clear(t);
    mpz_clear(tmp);
}

void Schoenhage::recursive(mpz_t& a, mpz_t& b, mpz_t& c, unsigned int m, Matrix U) {
    size_t a_len = mpz_sizeinbase(a, 2);
    size_t b_len = mpz_sizeinbase(b, 2);
    size_t c_len = mpz_sizeinbase(c, 2);
    unsigned int n = std::max(a_len, std::max(b_len, c_len)) - m;
    bool is_low_step;

    if (n < 200) {
        nonrecursive(a, b, c, m, U);
        return;
    }

    mpz_t tmp1, tmp2, t, alpha, beta, gamma, a0, b0, c0;
    mpz_init(a0); mpz_init(b0); mpz_init(c0);
    mpz_init(tmp1); mpz_init(tmp2); mpz_init(t);
    mpz_init(alpha);
    mpz_init(beta);
    mpz_init(gamma);

    MatrixData<mpz_t> dU = U.data<mpz_t>();
    mpz_t& U00 = dU(0,0);
    mpz_t& U01 = dU(0,1);
    mpz_t& U10 = dU(1,0);
    mpz_t& U11 = dU(1,1);

    // Check if min(a, b, c) < 2**(m+2)
    if (std::min(std::min(a_len, b_len), c_len) <= m + 2) {
        // alpha, beta, gamma = a, b, c
        mpz_set(alpha, a);
        mpz_set(beta, b);
        mpz_set(gamma, c);
        U.set_identity();
    } else {
        // R2 choose n minimal such that a, b, c <= 2**(m+n)
        unsigned int m_prime;
        unsigned int p;
        if (m <= n) {
            m_prime = m;
            p = 0;
        } else {
            m_prime = n;
            p = m + 1 - n;

            // a, b, c holds the high bits
            // a0, b0, c0 holds the low bits
            mpz_tdiv_r_2exp(a0, a, p);
            mpz_tdiv_r_2exp(b0, b, p);
            mpz_tdiv_r_2exp(c0, c, p);
            
            mpz_tdiv_q_2exp(a, a, p);
            mpz_tdiv_q_2exp(b, b, p);
            mpz_tdiv_q_2exp(c, c, p);

            a_len = mpz_sizeinbase(a, 2);
            b_len = mpz_sizeinbase(b, 2);
            c_len = mpz_sizeinbase(c, 2);
        }

        // R3
        unsigned int h = m_prime + (n / 2);
        if (std::min(a_len, std::min(b_len, c_len)) <= h) {
            // x, y, z synonymous with a, b, c
            U.set_identity();
        } else {
            // R4
            recursive(a, b, c, h, U);
            a_len = mpz_sizeinbase(a, 2);
            b_len = mpz_sizeinbase(b, 2);
            c_len = mpz_sizeinbase(c, 2);
        }

        // R5
        bool going_to_r8 = false;
        while (std::max(a_len, std::max(b_len, c_len)) > h) {
            if (is_minimal(a, b, c, m_prime)) {
                mpz_set(alpha, a);
                mpz_set(beta, b);
                mpz_set(gamma, c);
                going_to_r8 = true;
                break;
            } else {
                // One simple step on a, b, c above 2^m'
                // Update U
                simple_step(a, b, c, m_prime, t, is_low_step);
                a_len = mpz_sizeinbase(a, 2);
                b_len = mpz_sizeinbase(b, 2);
                c_len = mpz_sizeinbase(c, 2);
                if (is_low_step) {
                    // low step
                    // We must right multiply U by
                    // [1 -(-t)]
                    // [0  1]
                    // That is, add t copies of U0 to U1
                    mpz_mul(tmp1, U00, t);
                    mpz_add(U01, U01, tmp1);
                    mpz_mul(tmp1, U10, t);
                    mpz_add(U11, U11, tmp1);
                } else {
                    // high step
                    // We must right multiply U by
                    // [1  0]
                    // [-t 1]
                    mpz_mul(tmp1, U01, t);
                    mpz_sub(U00, U00, tmp1);
                    mpz_mul(tmp1, U11, t);
                    mpz_sub(U10, U10, tmp1);
                }
            }
        }

        if (!going_to_r8) {
            // R6
            Matrix U_prime(ElementType::MPZ, 2, 2);

            recursive(a, b, c, m_prime, U_prime);
            mpz_set(alpha, a);
            mpz_set(beta, b);
            mpz_set(gamma, c);

            // R7
            MatrixData<mpz_t> dU_prime = U_prime.data<mpz_t>();
            // Use t, tmp1, tmp2 as temporary variables for matmul
            // U = U * U_prime
            mpz_mul(tmp1, U00, dU_prime(0,0));
            mpz_mul(tmp2, U01, dU_prime(1,0));
            mpz_add(t, tmp1, tmp2);
            mpz_mul(tmp1, U00, dU_prime(0,1));
            mpz_mul(tmp2, U01, dU_prime(1,1));
            mpz_set(U00, t);
            mpz_add(U01, tmp1, tmp2);

            mpz_mul(tmp1, U10, dU_prime(0,0));
            mpz_mul(tmp2, U11, dU_prime(1,0));
            mpz_add(t, tmp1, tmp2);
            mpz_mul(tmp1, U10, dU_prime(0,1));
            mpz_mul(tmp2, U11, dU_prime(1,1));
            mpz_set(U10, t);
            mpz_add(U11, tmp1, tmp2);
        }

        // R8
        if (p > 0) {
            // Apply U to [a0, b0, c0]
            // [U00 U10]   [a0 b0/2]   [U00 U01]
            // [U01 U11] . [b0/2 c0] . [U10 U11]

            Matrix M(ElementType::MPZ, 2, 2);
            MatrixData<mpz_t> dM = M.data<mpz_t>();
            mpz_mul_2exp(dM(0,0), a0, 1);
            mpz_set(dM(0,1), b0);
            mpz_set(dM(1,0), b0);
            mpz_mul_2exp(dM(1,1), c0, 1);

            MatrixMultiplication mm1(M, M, U, cc);
            mm1.solve();
            MatrixMultiplication mm2(M, U.transpose(), M, cc);
            mm2.solve();

            mpz_tdiv_q_2exp(a0, dM(0,0), 1);
            mpz_set(b0, dM(0,1));
            mpz_tdiv_q_2exp(c0, dM(1,1), 1);

            mpz_mul_2exp(alpha, alpha, p);
            mpz_add(alpha, alpha, a0);

            mpz_mul_2exp(beta, beta, p);
            mpz_add(beta, beta, b0);

            mpz_mul_2exp(gamma, gamma, p);
            mpz_add(gamma, gamma, c0);
        }
    }

    mpz_set(a, alpha);
    mpz_set(b, beta);
    mpz_set(c, gamma);

    // R9
    while (!is_minimal(a, b, c, m)) {
        simple_step(a, b, c, m, t, is_low_step);
        if (mpz_cmp_si(t, 0) == 0) {
            // t = 0
            break;
        } else if (is_low_step) {
            // low step
            // We must right multiply U by
            // [1 -(-t)]
            // [0  1]
            // That is, add t copies of U0 to U1
            mpz_mul(tmp1, U00, t);
            mpz_add(U01, U01, tmp1);
            mpz_mul(tmp1, U10, t);
            mpz_add(U11, U11, tmp1);
        } else {
            // high step
            // We must right multiply U by
            // [1  0]
            // [-t 1]
            mpz_mul(tmp1, U01, t);
            mpz_sub(U00, U00, tmp1);
            mpz_mul(tmp1, U11, t);
            mpz_sub(U10, U10, tmp1);
        }
    }

    mpz_clear(tmp1); mpz_clear(tmp2); mpz_clear(t);
    mpz_clear(a0); mpz_clear(b0); mpz_clear(c0);
    mpz_clear(alpha);
    mpz_clear(beta);
    mpz_clear(gamma);
}

void Schoenhage::solve() {
    log_start();

    if (n == 0) {
        U.set_identity();
        log_end();
        return;
    }

    mpz_t a, b, c;
    unsigned int m = 0; // s = 2**m = 1
    mpz_t tmp, det2;

    mpz_init(a);
    mpz_init(b);
    mpz_init(c);
    mpz_init(tmp);
    mpz_init(det2);

    mpz_set_ui(a, 0);
    mpz_set_ui(b, 0);
    mpz_set_ui(c, 0);

    // Calculate a = <b0, b0>
    MatrixData<mpz_t> dB = M.data<mpz_t>();
    for (unsigned int i = 0; i < this->m; i++) {
        mpz_mul(tmp, dB(i, 0), dB(i, 0));
        mpz_add(a, a, tmp);
    }

    if (n == 1) {
        long exp;
        double d = mpz_get_d_2exp(&exp, a);
        params.L.profile[0] = (exp + log2(fabs(d))) / 2;
        U.set_identity();
        mpz_clear(a);
        mpz_clear(b);
        mpz_clear(c);
        mpz_clear(tmp);
        mpz_clear(det2);
        log_end();
        return;
    }

    // Calculate b = 2*<b0, b1>, c = <b1, b1>
    for (unsigned int i = 0; i < this->m; i++) {
        mpz_mul(tmp, dB(i, 0), dB(i, 1));
        mpz_add(b, b, tmp);
        mpz_mul(tmp, dB(i, 1), dB(i, 1));
        mpz_add(c, c, tmp);
    }
    mpz_mul_2exp(b, b, 1);

    // determinant = b1[0] * b2[1] - b2[0] * b1[1]
    // b^2/4 - ac = <b1, b2>**2 - <b1,b1><b2,b2>
    // = (b1[0] * b2[0] + b1[1] * b2[1])**2 - (b1[0] * b1[0] + b1[1]*b1[1]) * (b2[0] * b2[0] + b2[1] * b2[1])
    // = b10*b10*b20*b20 + 2*b10*b20*b11*b21 + b11*b11*b21*b21 - b10b10b20b20 - b10b10b21b21 - b11b11b20b20 - b11b11b21b21
    // = 2*b10*b20*b11*b21 - b10b10b21b21 - b11b11b20b20
    // = -(b10b21 - b11b20)^2
    // Thus abs(b^2 - 4*a*c) = 4*abs(det)^2
    // Let's use tmp and det2 to calculate 4*det^2
    mpz_mul(det2, b, b);
    mpz_mul(tmp, a, c);
    mpz_mul_2exp(tmp, tmp, 2);
    mpz_sub(det2, det2, tmp); // det2 = b^2 - 4ac

    MatrixData<mpz_t> dU = U.data<mpz_t>();
    if (mpz_sgn(det2) == 0) {
        // The input is linearly dependent. Handle this edge case.
        bool rank_zero = false;
        // Zero out U
        for (unsigned int i = 0; i < dU.nrows(); i++) {
            for (unsigned int j = 0; j < dU.ncols(); j++) {
                mpz_set_ui(dU(i,j), 0);
            }
        }

        if (mpz_sgn(a) == 0 && mpz_sgn(c) == 0) {
            // Both columns are zero vectors. If we do nothing, this will be handled later.
            rank_zero = true;
        } else if (mpz_sgn(a) == 0 && mpz_sgn(c) != 0) {
            // The first vector is all zeros and the second is nonzero.
            mpz_set_ui(dU(1,0), 1); // First vector in output is second vector in input
        } else if (mpz_sgn(a) != 0 && mpz_sgn(c) == 0) {
            // The first vector is nonzero and the second is all zeros.
            mpz_set_ui(dU(0,0), 1); // First vector in output is first vector in input
        } else {
            // Both columns are nonzero, but they are linearly dependent.
            // We need to do an extended GCD computation.

            // We could just do a GCD computation on 2*<b0, b0>, 2*<b0, b1>
            // which is just 2*a and b
            mpz_mul_2exp(a, a, 1);
            xgcd(a, b, tmp, dU(0,0), dU(1,0));
        }

        // Apply U to M
        MatrixMultiplication mm(M.submatrix(0, M.nrows(), 0, U.ncols()), M, U, cc);
        mm.solve();

        MatrixData<mpz_t> dM = M.data<mpz_t>();
        for (unsigned int j = U.ncols(); j < M.ncols(); j++) {
            // In case U is narrower than M (because we know the lattice is low rank)
            for (unsigned int i = 0; i < M.nrows(); i++) {
                mpz_set_ui(dM(i,j), 0);
            }
        }

        if (!rank_zero) {
            // To update the profile, we need to know how long the new vector is.
            MatrixData<mpz_t> dM = M.data<mpz_t>();
            mpz_set_ui(det2, 0);
            for (unsigned int i = 0; i < this->m; i++) {
                mpz_mul(tmp, dM(i, 0), dM(i, 0));
                mpz_add(det2, det2, tmp);
            }

            double dbl_d;
            long d_exp;
            dbl_d = mpz_get_d_2exp(&d_exp, det2);

            params.L.profile[0] = (d_exp + log2(dbl_d)) / 2;
            mon->profile_update(
                &params.L.profile[0],
                profile_offset, offset, offset + 1);
        }

        mpz_clear(a);
        mpz_clear(b);
        mpz_clear(c);
        mpz_clear(tmp);
        mpz_clear(det2);
        log_end();
        return;
    }

    bool flipped_b;
    // a and c are guaranteed to be positive, as they are
    // norms of nonzero vectors.
    // In the case where c is negative, flip the signs of
    // B[1], so <B[0], B[1]> is negative.
    // Later we will have to flip the signs in U to account
    // for this change
    if (mpz_sgn(b) <= 0) {
        flipped_b = true;
        mpz_neg(b, b);
    } else {
        flipped_b = false;
    }

    if (mpz_sgn(b) > 0) {
        // If it's 0, then B[0] and B[1] are perpendicular,
        // so we don't need to do this recursive step
        assert(mpz_cmp_ui(a, 0) > 0);
        assert(mpz_sgn(b) > 0);
        assert(mpz_cmp_ui(c, 0) > 0);

        recursive(a, b, c, m, U);
    } else {
        U.set_identity();
    }

    // minimality over 1 is actually a different condition from Lagrange reduction.
    // minimality gives
    //   ||b0|| >= 1
    //   ||b1|| >= 1
    //   <b0, b1> >= 1
    //   b0 == b1 OR
    //   <b0, b1 - b0> < 1 AND <b1, b0 - b1> < 1

    // What we want for Lagrange reduction is
    //   ||b0|| <= ||b1||
    //   2*<b0,b1> <= ||b0||^2
    //   a <= c && |b| <= a
    // c <- c - b + a
    
    // The first property is easy to satisfy, since we just optionally flip a and c
    // so a is smaller
    //mpfr_printf("a %Zd\nb %Zd\nc %Zd\n", a, b, c);
    if (mpz_cmp(c, a) < 0) {
        mpz_swap(a, c);
        mpz_swap(dU(0,0), dU(0,1));
        mpz_swap(dU(1,0), dU(1,1));
    }
    //mpfr_printf("a %Zd\nb %Zd\nc %Zd\n", a, b, c);
    if (mpz_cmpabs(b, a) > 0) {
        // Not size reduced
        // Do a low step.
        mpz_sub(b, b, a);
        mpz_sub(c, c, b);
        mpz_sub(b, b, a);

        mpz_sub(dU(0,1), dU(0,1), dU(0,0));
        mpz_sub(dU(1,1), dU(1,1), dU(1,0));
        // Might have to swap again, but the we are guaranteed to be done.
        if (mpz_cmp(c, a) < 0) {
            mpz_swap(a, c);
            mpz_swap(dU(0,0), dU(0,1));
            mpz_swap(dU(1,0), dU(1,1));
        }
    
    }
    assert(mpz_cmp(a, c) <= 0);
    assert(mpz_cmpabs(b, a) <= 0);

    if (flipped_b) {
        mpz_neg(dU(1,0), dU(1,0));
        mpz_neg(dU(1,1), dU(1,1));
    }

    // a is ||b1||^2
    // b is 2*<b1,b2>
    // c is ||b2||^2
    double dbl_a, dbl_d;
    long a_exp, d_exp;
    dbl_a = mpz_get_d_2exp(&a_exp, a);

    params.L.profile[0] = (a_exp + log2(dbl_a)) / 2;
    dbl_d = mpz_get_d_2exp(&d_exp, det2);
    double log_det = ((d_exp + log2(fabs(dbl_d))) - 2) / 2;

    params.L.profile[1] = log_det - (a_exp + log2(fabs(dbl_a))) / 2;
    mon->profile_update(
        &params.L.profile[0],
        profile_offset, offset, offset + 2);

    // Apply U to M
    MatrixMultiplication mm(M, M, U, cc);
    mm.solve();

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
    mpz_clear(tmp);
    mpz_clear(det2);

    log_end();
}

}
}