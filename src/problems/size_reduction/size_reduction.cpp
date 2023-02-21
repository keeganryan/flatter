#include "problems/size_reduction.h"

#include <cassert>

#include "blocked.h"
#include "elementary_ZZ.h"
#include "elementary_ll.h"

namespace flatter {

SizeReduction::SizeReduction() {
    _is_configured = false;
}

SizeReduction::SizeReduction(const Matrix& R, const Matrix& U,
                                    const ComputationContext& cc) :
    Base(R, U, cc) 
{
    _is_configured = false;
    configure(R, U, cc);
}

SizeReduction::~SizeReduction() {
    if (_is_configured) {
        unconfigure();
    }
}

void SizeReduction::unconfigure() {
    assert(_is_configured);

    delete szred;

    _is_configured = false;
}

void SizeReduction::configure(const Matrix& R, const Matrix& U,
                                 const ComputationContext& cc) {
    if (_is_configured) {
        unconfigure();
    }
    assert(R.ncols() == R.nrows());

    Base::configure(R, U, cc);
    if (R.type() == ElementType::INT64 && U.type() == ElementType::INT64) {
        szred = new SizeReductionImpl::ElementaryLL(R, U, cc);
    } else if (n <= 100) {
        szred = new SizeReductionImpl::ElementaryZZ(R, U, cc);
    } else {
        szred = new SizeReductionImpl::Blocked(R, U, cc);
    }

    _is_configured = true;
}

void SizeReduction::solve() {
    assert(_is_configured);

    bool use_tasks = cc.is_threaded();

    if (use_tasks && omp_get_active_level() == 0) {
        #pragma omp parallel num_threads(cc.nthreads()) if (use_tasks)
        {
            #pragma omp single
            {
                szred->solve();
            }
        }
    } else {
        #pragma omp taskgroup
        {
            szred->solve();
        }
    }
}

}