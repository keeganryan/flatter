#include "sublattice_split_2.h"

#include <cassert>
#include <cstdio>

namespace flatter {

SubSplitPhase2::SubSplitPhase2(unsigned int n) {
    all = nullptr;
    left = right = nullptr;

    if (n > 1) {
        k = next_smaller(n);
        left = new SubSplitPhase2(k);
        right = new SubSplitPhase2(n - k);
        all = new SubSplitPhase3(left->all, right->all);
    } else {
        all = new SubSplitPhase3(1);
    }
    this->n = n;

    iter = 0;
}

SubSplitPhase2::~SubSplitPhase2() {
    if (n > 1) {
        delete left;
        delete right;
        all->left_child = nullptr;
        all->right_child = nullptr;
        delete all;
    } else {
        delete all;
    }
}

std::vector<sublattice> SubSplitPhase2::get_sublattices () {
    std::vector<sublattice> ret;

    sublattice s;

    if (n == 3) {
        if (iter == 0) {
            s.start = 0;
            s.end = k;
        } else {
            s.start = 0;
            s.end = n;
        }
    } else {
        if (iter == 0) {
            s.start = 0;
            s.end = k;
        } else if (iter == 1) {
            s.start = k;
            s.end = n;
        } else {
            s.start = 0;
            s.end = n;
        }
    }
    ret.push_back(s);

    SublatticeSplit* c = this->get_child_split(0);
    if (c->n != s.end - s.start) {
        assert(false);
    }

    return ret;
}

void SubSplitPhase2::advance_sublattices() {
    iter += 1;
}

SublatticeSplit* SubSplitPhase2::get_child_split(unsigned int i) {
    assert(i == 0);

    if (n == 3) {
        if (iter == 0) {
            return left;
        } else {
            return all;
        }
    } else {
        if (iter == 0) {
            assert(left != nullptr);
            return left;
        } else if (iter == 1) {
            assert(right != nullptr);
            return right;
        } else {
            return all;
        }
    }
}

bool SubSplitPhase2::stopping_point() {
    if (n == 3) {
        if (iter <= 1) {
            return false;
        }
    } else {
        if (iter <= 2) {
            return false;
        }
    }
    return true;
}

unsigned int SubSplitPhase2::next_smaller(unsigned int n) {
    assert (n >= 2);
    if (n == 3) {
        return 2;
    }
    return n / 2;
}

}