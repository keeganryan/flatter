#include "eigen_impl.h"
#include "workspace_buffer.h"

#include <cassert>
#include <Eigen/Dense>

namespace EigenLib = Eigen;

namespace flatter {
namespace QRFactorizationImpl {

const std::string Eigen::impl_name() {return "Eigen";}

Eigen::Eigen(const Matrix& A, const Matrix& tau, const Matrix& T,
                         const ComputationContext& cc) :
    Base(A, tau, T, cc)
{
    _is_configured = false;
    configure(A, tau, T, cc);
}

Eigen::~Eigen() {
    if (_is_configured) {
        unconfigure();
    }
}

void Eigen::unconfigure() {
    if (!_save_tau) {
        free(tau_ptr);
    }
    _is_configured = false;
}

void Eigen::configure(const Matrix& A, const Matrix& tau, const Matrix& T,
                            const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    assert(!A.is_transposed());
    assert(A.type() == ElementType::DOUBLE);
    assert(tau.type() == ElementType::DOUBLE);
    assert(T.nrows() == 0 || T.type() == ElementType::DOUBLE);

    Base::configure(A, tau, T, cc);

    dA = A.data<double>();

    if (this->tau.nrows() == 0) {
        _save_tau = false;
    } else {
        _save_tau = true;
        MatrixData<double> dtau = tau.data<double>();
        tau_ptr = dtau.get_data();
    }

    if (T.nrows() == 0) {
        _save_block_reflector = false;
    } else {
        dT = T.data<double>();
        _save_block_reflector = true;
    }
    assert(!_save_block_reflector);

    if (!_save_tau) {
        this->tau_ptr = new double[rank];
    }

    _is_configured = true;
}

void Eigen::solve() {
    log_start();

    unsigned int m = dA.nrows();
    unsigned int n = dA.ncols();
    EigenLib::MatrixXd A(m, n);
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++) {
            A(i, j) = dA(i, j);
        }
    }
    auto QR = A.householderQr();
    EigenLib::MatrixXd R = QR.matrixQR();
    EigenLib::VectorXd h = QR.hCoeffs();

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++) {
            dA(i, j) = R(i, j);
            if (i == j) {
                assert(std::isfinite(dA(i,i)));
            }
        }
    }
    for (unsigned int i = 0; i < n; i++) {
        tau_ptr[i] = h(i);
    }

    if (!_save_tau && !_save_block_reflector) {
        clear_subdiagonal();
    }

    log_end();
}

void Eigen::clear_subdiagonal() {
    // Clear below diagonal
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < i && j < n; j++) {
            this->dA(i,j) = 0;
        }
    }
}

}
}