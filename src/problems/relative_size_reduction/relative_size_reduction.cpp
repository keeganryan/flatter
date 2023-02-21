#include "problems/relative_size_reduction.h"

#include <cassert>

#include "generic.h"
#include "triangular.h"
#include "orthogonal.h"
#include "orthogonal_double.h"

namespace flatter {

RelativeSizeReduction::RelativeSizeReduction() {
    _is_configured = false;
}

RelativeSizeReduction::RelativeSizeReduction(const RelativeSizeReductionParams& params,
                               const ComputationContext& cc) :
    Base(params, cc) 
{
    _is_configured = false;
    configure(params, cc);
}

RelativeSizeReduction::~RelativeSizeReduction() {
    if (_is_configured) {
        unconfigure();
    }
}

void RelativeSizeReduction::unconfigure() {
    assert(_is_configured);

    delete szred2;

    _is_configured = false;
}

void RelativeSizeReduction::configure(const RelativeSizeReductionParams& params,
                                 const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    
    Base::configure(params, cc);

    if (params.is_B1_upper_triangular) {
        szred2 = new RelativeSizeReductionImpl::Triangular(params,cc);
    } else if (params.RV.nrows() != 0) {
        if (params.RV.type() == ElementType::MPFR) {
            szred2 = new RelativeSizeReductionImpl::Orthogonal(params, cc);
        } else if (params.RV.type() == ElementType::DOUBLE) {
            szred2 = new RelativeSizeReductionImpl::OrthogonalDouble(params, cc);
        } else {
            assert(0);
        }
    } else {
        szred2 = new RelativeSizeReductionImpl::Generic(params, cc);
    }

    _is_configured = true;
}

void RelativeSizeReduction::solve() {
    assert(_is_configured);

    bool use_tasks = cc.is_threaded();

    if (use_tasks && omp_get_active_level() == 0) {
        #pragma omp parallel num_threads(cc.nthreads()) if (use_tasks)
        {
            #pragma omp single
            {
                szred2->solve();
            }
        }
    } else {
        #pragma omp taskgroup
        {
            szred2->solve();
        }
    }
}

}