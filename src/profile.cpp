#include "data/lattice/profile.h"

#include <cassert>
#include <cmath>

namespace flatter {

Profile::Profile() {
    is_valid_ = false;
    n = 0;
}

Profile::Profile(unsigned int n) {
    if (n == 0) {
        is_valid_ = false;
        this->n = 0;
    } else {
        profile_elems = std::shared_ptr<double[]>(new double[n]);
        for (unsigned int i = 0; i < n; i++) {
            profile_elems[i] = nan("");
        }
        is_valid_ = true;
        this->n = n;
    }
}

bool Profile::is_valid() const {
    return is_valid_;
}

double& Profile::operator[](unsigned int i) {
    assert(is_valid_);
    assert(i < n);
    return profile_elems[i];
}

const double& Profile::operator[](unsigned int i) const {
    assert(is_valid_);
    assert(i < n);
    return profile_elems[i];
}

Profile Profile::subprofile(unsigned int start, unsigned int end) {
    assert(is_valid_);
    assert(start < end);
    assert(end <= n);

    Profile p_sub(end - start);

    for(unsigned int i = 0; i < end - start; i++) {
        p_sub[i] = profile_elems[start + i];
    }
    return p_sub;
}

double Profile::get_drop() const {
    assert (is_valid_);

    // Calculate how many bits are needed
    double *max_from_left = new double[n];
    double *min_from_right = new double[n];

    max_from_left[0] = profile_elems[0];
    min_from_right[n - 1] = profile_elems[n - 1];
    for (unsigned int i = 0; i < n - 1; i++) {
        max_from_left[i + 1] = std::max(profile_elems[i + 1], max_from_left[i]);
        min_from_right[n - i - 2] = std::min(profile_elems[n - i - 2], min_from_right[n - i - 1]);
    }

    double spread = max_from_left[n - 1] - min_from_right[0];
    for (unsigned int i = 0; i < n - 1; i++) {
        if (min_from_right[i+1] > max_from_left[i]) {
            spread -= (min_from_right[i+1] - max_from_left[i]);
        }
    }

    delete[] max_from_left;
    delete[] min_from_right;

    return spread;
}

double Profile::get_spread() const {
    assert (is_valid_);

    // Calculate how many bits are needed
    double *max_from_left = new double[n];
    double *min_from_right = new double[n];

    max_from_left[0] = profile_elems[0];
    min_from_right[n - 1] = profile_elems[n - 1];
    for (unsigned int i = 0; i < n - 1; i++) {
        max_from_left[i + 1] = std::max(profile_elems[i + 1], max_from_left[i]);
        min_from_right[n - i - 2] = std::min(profile_elems[n - i - 2], min_from_right[n - i - 1]);
    }

    double spread = max_from_left[n - 1] - min_from_right[0];

    delete[] max_from_left;
    delete[] min_from_right;

    return spread;
}

}