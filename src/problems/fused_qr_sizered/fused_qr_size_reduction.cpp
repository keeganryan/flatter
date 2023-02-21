#include "problems/fused_qr_size_reduction.h"

#include <cassert>

#include "columnwise.h"
#include "columnwise_double.h"
#include "iterated.h"
#include "lazy_refine.h"
#include "seysen_refine.h"

namespace flatter {

FusedQRSizeReduction::FusedQRSizeReduction() {
    _is_configured = false;
}

FusedQRSizeReduction::FusedQRSizeReduction(
    const Matrix& B, const Matrix& R, const Matrix& U,
    const ComputationContext& cc) :
    FusedQRSizeReduction(FusedQRSizeReductionParams(B, R, U), cc)
{}

FusedQRSizeReduction::FusedQRSizeReduction(
    const FusedQRSizeReductionParams& params,
    const ComputationContext& cc) :
    Base(params, cc)
{
    _is_configured = false;
    configure(params, cc);
}

FusedQRSizeReduction::~FusedQRSizeReduction() {
    if (_is_configured) {
        unconfigure();
    }
}

void FusedQRSizeReduction::unconfigure() {
    assert(_is_configured);
    delete this->fqrszred;
    _is_configured = false;
}
void FusedQRSizeReduction::configure(
    const Matrix& B,
    const Matrix& R, const Matrix& U,
    const ComputationContext& cc) {
    FusedQRSizeReductionParams params(B, R, U);
    this->configure(params, cc);
}

void FusedQRSizeReduction::configure(
    const FusedQRSizeReductionParams& params,
    const ComputationContext& cc) {

    if (_is_configured) {
        unconfigure();
    }
    Base::configure(params, cc);

    if (params.prereduced_sublattice_inds.size() > 0) {
        this->fqrszred = new FusedQRSizeRedImpl::LazyRefine(params, cc);
    } else {
        if (R.type() == ElementType::MPFR) {
            this->fqrszred = new FusedQRSizeRedImpl::Columnwise(params, cc);
        } else if (R.type() == ElementType::DOUBLE) {
            this->fqrszred = new FusedQRSizeRedImpl::ColumnwiseDouble(params, cc);
        } else {
            assert(0);
        }
    }

    _is_configured = true;
}

void FusedQRSizeReduction::solve() {
    assert(_is_configured);
    bool use_tasks = cc.is_threaded();

    if (use_tasks && omp_get_active_level() == 0) {
        #pragma omp parallel num_threads(cc.nthreads()) if (use_tasks)
        {
            #pragma omp single
            {
                fqrszred->solve();
            }
        }
    } else {
        #pragma omp taskgroup
        {
            fqrszred->solve();
        }
    }
}

}