#include "sublattice_split_3.h"

#include <cassert>
#include <cstdio>

namespace flatter {

SubSplitPhase3::SubSplitPhase3(unsigned int n) :
    SubSplitPhase3(n, n/2)
{}

SubSplitPhase3::SubSplitPhase3(unsigned int n, unsigned int k) :
    SublatticeSplit(n)
{
    this->k = k;

    left_child = nullptr;
    right_child = nullptr;
    mid_child = nullptr;

    if (n == 3) {
        this->k = 2;
        left_child = new SubSplitPhase3(2);
        right_child = new SubSplitPhase3(1);
        mid_child = new SubSplitPhase3(left_child->right_child, right_child);
    } else if (n >= 2) {
        left_child = new SubSplitPhase3(k);
        right_child = new SubSplitPhase3(n - k);
        mid_child = new SubSplitPhase3(left_child->right_child, right_child->left_child);
    }
    iter = 0;
}

SubSplitPhase3::SubSplitPhase3(SubSplitPhase3* l, SubSplitPhase3* r) {
    left_child = l;
    right_child = r;
    mid_child = nullptr;

    k = (l == nullptr) ? 1 : l->n;
    n = k + ((r == nullptr) ? 1 : r->n);
    iter = 0;

    if (n == 3) {
        if (l->n == 2) {
            mid_child = new SubSplitPhase3(left_child->right_child, right_child);
        } else {
            mid_child = new SubSplitPhase3(left_child, right_child->left_child);
        }
    } else if (left_child != nullptr &&
        left_child->right_child != nullptr &&
        right_child != nullptr &&
        right_child->left_child != nullptr) {
        mid_child = new SubSplitPhase3(left_child->right_child, right_child->left_child);
    }
}

SubSplitPhase3::~SubSplitPhase3() {
    // Children of middle child will be deleted by other children
    if (mid_child != nullptr) {
        mid_child->left_child = nullptr;
        mid_child->right_child = nullptr;
        delete mid_child;
    }
    if (left_child != nullptr) {
        delete left_child;
    }
    if (right_child != nullptr) {
        delete right_child;
    }
}

std::vector<sublattice> SubSplitPhase3::get_sublattices () {
    std::vector<sublattice> ret;

    sublattice s1;
    sublattice s2;
    if (n != 3) {
        if (iter % 2 == 0) {
            s1.start = left_child->k;
            s1.end = k + right_child->k;
            ret.push_back(s1);
        } else {
            s1.start = 0;
            s1.end = k;
            ret.push_back(s1);
            s2.start = k;
            s2.end = n;
            ret.push_back(s2);
        }
    } else {
        if (iter % 2 == 0) {
            s1.start = 0;
            s1.end = 2;
        } else {
            s1.start = 1;
            s1.end = 3;
        }
        ret.push_back(s1);
    }

    return ret;
}

void SubSplitPhase3::advance_sublattices() {
    left_child->reset();
    right_child->reset();
    iter += 1;
}

void SubSplitPhase3::reset() {
    iter = 0;
    if (left_child != nullptr) {
        left_child->reset();
    }
    if (right_child != nullptr) {
        right_child->reset();
    }
}

unsigned int SubSplitPhase3::cycle_len() {
    if (n == 3) {
        return 2;
    } else {
        return 2;
    }
}

SublatticeSplit* SubSplitPhase3::get_child_split(unsigned int i) {
    if (n == 3) {
        assert(i == 0);

        if (k == 2) {
            if (iter % 2 == 0) {
                return left_child;
            } else {
                return mid_child;
            }
        } else {
            if (iter % 2 == 0) {
                return mid_child;
            } else {
                return right_child;
            }
        }
    }
    
    assert (i < 2);
    if (iter % 2 == 0) {
        return mid_child;
    } else {
        if (i == 0) {
            return left_child;
        } else {
            return right_child;
        }
    }
}

bool SubSplitPhase3::stopping_point() {
    if (n == 3) {
        if (iter % 2 == 0) {
            return true;
        }
    } else {
        if (iter % 2 == 0) {
            return true;
        }
    }
    return false;
}

}