#include "problems/relative_size_reduction/base.h"

#include <cassert>
#include <sstream>

namespace flatter {
namespace RelativeSizeReductionImpl {

const std::string Base::prob_name() {return "Size Reduction 2";}
const std::string Base::impl_name() {return "Base Implementation";}
const std::string Base::param_headers() {return "d:3 P:-1";}
std::string Base::get_param_values() {
    std::stringstream ss;
    ss << this->n;
    return ss.str(); 
}

Base::Base() {}

Base::Base(const RelativeSizeReductionParams& params,
           const ComputationContext& cc) {
    Base::configure(params, cc);
}

void Base::configure(const RelativeSizeReductionParams& params,
                     const ComputationContext& cc) {
    this->params = params;

    this->n = params.B1.nrows();
    
    this->cc = cc;
}

}
}