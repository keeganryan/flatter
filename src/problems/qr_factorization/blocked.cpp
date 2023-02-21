#include "blocked.h"

#include <cassert>

#include "problems/matrix_multiplication.h"
#include "problems/qr_factorization.h"

namespace flatter {
namespace QRFactorizationImpl {

const std::string Blocked::impl_name() {return "Blocked";}

Blocked::Blocked(const Matrix& A, const Matrix& tau, const Matrix& T,
                 const ComputationContext& cc) :
    Base(A, tau, T, cc)
{
    _is_configured = false;
    configure(A, tau, T, cc);
}

Blocked::~Blocked() {
    if (_is_configured) {
        unconfigure();
    }
}

void Blocked::unconfigure() {
    if (!_save_block_reflector) {
        wsb->wfree(T_sub_1.get_data(), r1*r1);
        wsb->wfree(T_sub_0.get_data(), r0*r0);
    }

    if (!_save_tau) {
        tau = Matrix();
    }
    wsb->wfree(work, lwork);
    delete wsb;
    
    _is_configured = false;
}

void Blocked::configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                        const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }

    assert(!A.is_transposed());
    assert(A.type() == ElementType::MPFR);
    assert(tau.type() == ElementType::MPFR);
    assert(T.type() == ElementType::MPFR);
    assert(!cc.is_threaded());

    Base::configure(A, tau, T, cc);

    assert(rank > 1);
    dA = A.data<mpfr_t>();
    dT = T.data<mpfr_t>();

    r0 = (rank + 1) / 2; // Columns solved by step 0
    r1 = rank - r0; // Columns solved by step 1

    lwork = 6; // For larf, larfg
    alloc_size = lwork;
    if (this->tau.nrows() == 0) {
        _save_tau = false;
        alloc_size += rank;
    } else {
        MatrixData<mpfr_t> dtau = tau.data<mpfr_t>();
        tau_ptr = dtau.get_data();
        _save_tau = true;
    }

    if (T.nrows() == 0) {
        _save_block_reflector = false;
        alloc_size += r0 * r0 + r1 * r1;
    } else {
        _save_block_reflector = true;
    }
    
    wsb = new WorkspaceBuffer<mpfr_t>(alloc_size, prec);
    work = wsb->walloc(lwork);
    if (!_save_tau) {
        this->tau = Matrix(ElementType::MPFR, rank, 1, prec);
    }

    if (!_save_block_reflector) {
        // Todo: We could reuse T here
        T_sub_0 = MatrixData<mpfr_t>(wsb->walloc(r0 * r0), r0, r0);
        T_sub_1 = MatrixData<mpfr_t>(wsb->walloc(r1 * r1), r1, r1);
    } else {
        T_sub_0 = dT.submatrix(0, r0, 0, r0);
        T_sub_1 = dT.submatrix(r0, n, r0, n);
    }

    MatrixData<mpfr_t> panel = dA.submatrix(0, A.nrows(), 0, r0);
    qr0.configure(panel, this->tau.submatrix(0, r0, 0, 1), T_sub_0, this->cc);

    MatrixData<mpfr_t> A2 = dA.submatrix(r0, m, r0, n);
    qr1.configure(A2, this->tau.submatrix(r0, n, 0, 1), T_sub_1, this->cc);

    _is_configured = true;
}

void Blocked::solve() {
    log_start();

    unsigned int bs = r0;

    WorkspaceBuffer<mpfr_t> wsb(1 + bs * bs + rank + 2*m*m + 2*m*bs + (m)*(n-bs) + (n-bs)*(n-bs) + 2*(bs)*(n-bs), prec);

    mpfr_rnd_t rnd = mpfr_get_default_rounding_mode();

    // QR-factor the left half of the matrix
    qr0.solve();

    // We have T and V, so construct Q
    mpfr_t* p_ONE = wsb.walloc(1);
    mpfr_t& ONE = *p_ONE;
    mpfr_set_ui(ONE, 1, rnd);

    MatrixData<mpfr_t> Q(wsb.walloc(m*m),m,m);
    MatrixData<mpfr_t> V(wsb.walloc(m*r0), m, r0);
    MatrixData<mpfr_t> Tmp1(wsb.walloc(r0*m),r0,m);
    MatrixData<mpfr_t> Tmp2(wsb.walloc(m*(n-r0)),m,(n-r0));

    MatrixData<mpfr_t> V1(wsb.walloc((m)*(n-bs)), m, n-bs);
    MatrixData<mpfr_t> Tmp3(wsb.walloc(bs*(n-bs)), bs, n-bs);
    MatrixData<mpfr_t> Tmp4(wsb.walloc(bs*(n-bs)), bs, n-bs);
    // Get V, T as a rectangular matrix
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < bs; j++) {
            if (j < i && i < bs) {
                mpfr_set_zero(T_sub_0(i, j), 0);
            }
            if (j < i) {
                mpfr_set(V(i, j), dA(i, j), rnd);
            } else if (j == i) {
                mpfr_set(V(i, j), ONE, rnd);
            } else {
                mpfr_set_zero(V(i, j), 0);
            }
        }
    }
    MatrixMultiplication mm1(Tmp1, T_sub_0, V.transpose(), cc);
    mm1.solve();
    MatrixMultiplication mm2(Q, V, Tmp1, cc);
    mm2.solve();
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < m; j++) {
            mpfr_neg(Q(i, j), Q(i, j), rnd);
            if (i == j) {
                mpfr_add(Q(i, j), Q(i, j), ONE, rnd);
            }
        }
    }

    MatrixMultiplication mm3(Tmp2, Q.transpose(), A.submatrix(0, m, bs, n), cc);
    mm3.solve();

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n - bs; j++) {
            mpfr_set(dA(i, bs+j), Tmp2(i, j), rnd);
        }
    }

    // QR reduce bottom right corner
    qr1.solve();


    if (_save_block_reflector) {
        // Get V1, T1 as a rectangular matrix
        for (unsigned int i = 0; i < m; i++) {
            for (unsigned int j = 0; j < n - bs; j++) {
                if (j < i && i < n - bs) {
                    mpfr_set_zero(T_sub_1(i, j), 0);
                }
                if (j + bs < i) {
                    mpfr_set(V1(i, j), dA(i, bs+j), rnd);
                } else if (j + bs == i) {
                    mpfr_set(V1(i, j), ONE, rnd);
                } else {
                    mpfr_set_zero(V1(i, j), 0);
                }
            }
        }

        MatrixMultiplication mm4(Tmp3, V.transpose(), V1, cc);
        mm4.solve();

        MatrixMultiplication mm5(Tmp4, Tmp3, T_sub_1, cc);
        mm5.solve();

        MatrixMultiplication mm6(Tmp3, T_sub_0, Tmp4, cc);
        mm6.solve();

        for (unsigned int i = 0; i < T.nrows(); i++) {
            for (unsigned int j = 0; j < T.ncols(); j++) {
                if (i < bs && j >= bs) {
                    // Top right
                    mpfr_neg(dT(i, j), Tmp3(i, j-bs), rnd);
                } 
                if (i >= bs && j < bs) {
                    // Bottom left
                    mpfr_set_zero(dT(i, j), 0);
                }
            }
        }
    }

    if (!_save_tau && !_save_block_reflector) {
        for (unsigned int i = 0; i < m; i++) {
            for (unsigned int j = 0; j < n; j++) {
                if (j < i) {
                    mpfr_set_zero(dA(i, j), 0);
                }
            }
        }
    }

    wsb.wfree(Tmp4.get_data(), bs*(n-bs));
    wsb.wfree(Tmp3.get_data(), bs*(n-bs));
    wsb.wfree(V1.get_data(), (m)*(n-bs));

    wsb.wfree(Tmp2.get_data(), m*(n-r0));
    wsb.wfree(Tmp1.get_data(), bs*m);
    wsb.wfree(V.get_data(), m*bs);
    wsb.wfree(Q.get_data(), m*m);
    wsb.wfree(p_ONE, 1);

    log_end();
}

}
}