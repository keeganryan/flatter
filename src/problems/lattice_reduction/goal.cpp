#include "problems/lattice_reduction/goal.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cmath>

namespace flatter {

LatticeReductionGoal::LatticeReductionGoal() {
    n = 0;
    quality = 0;
}

LatticeReductionGoal::LatticeReductionGoal(unsigned int n, double quality, bool proved) {
    this->proved = proved;
    assert (this->proved == false);
    this->n = n;
    this->quality = quality;
    this->best_slope = BKZ_BEST_SLOPE;
}

LatticeReductionGoal::LatticeReductionGoal( unsigned int n,
    double top_level_slope, double base_slope, double g, unsigned int top_N) {
        this->proved = true;
        this->n = n;
        if (top_N == 0) {
            this->top_N = n;
        } else {
            this->top_N = top_N;
        }
        this->top_slope = top_level_slope;
        this->best_slope = base_slope;
        this->g = g;
        this->log_g = log2(g);
        assert(top_level_slope > base_slope);
}

double LatticeReductionGoal::get_alpha_n(unsigned int n) {
    return best_slope + pow((double)n / (double)this->top_N, this->log_g) * (top_slope - best_slope);
}

bool LatticeReductionGoal::check(Profile profile) {
    assert(n != 0);
    //assert(quality != 0);
    if (n == 1) {
        return true;
    }

    double max_drop;
    double mu_sep;
    if (proved) {
        max_drop = get_alpha_n(n) * n;
        return profile.get_drop() < max_drop;
    } else {
        double lgn = log2(n);
        max_drop = quality * 3 * (1 + pow(3, lgn + 1) - pow(2, lgn + 2)) / 2;
        max_drop += best_slope * n;

        double gamma_i = quality * pow(3, lgn);
        mu_sep = (max_drop - gamma_i) / 2. + gamma_i;
    }

    unsigned n_L = n / 2;
    if (n == 3) {
        n_L = 2;
    }
    unsigned n_R = n - n_L;
    
    double lgnl = log2(n_L);
    double l_drop = best_slope*(n_L) + quality * 3 * (1 + pow(3, lgnl + 1) - pow(2, lgnl + 2)) / 2;
    double lgnr = log2(n - n_L);
    double r_drop = best_slope*(n - n_L) + quality * 3 * (1 + pow(3, lgnr + 1) - pow(2, lgnr + 2)) / 2;

    //assert(max_drop > l_drop + r_drop);
    //double my_drop = profile[n_L - 1] - profile[n_L];
    // We want to avoid the case where the entire drop is across this gap, because
    // then we could do better by continuing to reduce. We also want to return before
    // this drop is smaller than necessary, because then all drops will be an underestimate,
    // and we are achieving a better than requested approximation factor.
    // We will go between the two and check if
    //     my_drop < max_drop - 0.5 * (l_drop + r_drop)
    unsigned n1 = n_L / 2;
    if (n_L == 3) {
        n1 = 2;
    }
    unsigned n3 = n_R / 2;
    if (n_R == 3) {
        n3 = 2;
    }
    double mid_drop = profile.subprofile(n1, n_L + n3).get_drop();
    

    double mu_L = 0;
    double mu_R = 0;

    for (unsigned int i = 0; i < n_L; i++) {
        mu_L += profile[i];
    }
    mu_L /= (n_L);
    for (unsigned int i = n_L; i < n; i++) {
        mu_R += profile[i];
    }
    mu_R /= (n - n_L);

    //LatticeReductionGoal left = this->subgoal(0, n/2);
    //LatticeReductionGoal right = this->subgoal(n/2, n);

    bool self_satisfactory = 
        (profile.get_drop() < max_drop) && 
        (mu_L - mu_R < mu_sep) && 
        (mid_drop <= l_drop + (max_drop - l_drop - r_drop));
        //(my_drop <= (max_drop - 0.5*(l_drop + r_drop)));

    return self_satisfactory; // && left.check(profile.subprofile(0, n/2)) && right.check(profile.subprofile(n/2, n));
}

LatticeReductionGoal LatticeReductionGoal::subgoal(unsigned int start, unsigned int end) {
    assert(start < end);
    assert(end <= n);

    if (proved) {
        return LatticeReductionGoal(end - start, this->top_slope, this->best_slope, this->g, this->top_N);
    } else {
        LatticeReductionGoal new_goal(end - start, this->quality, false);
        new_goal.best_slope = this->best_slope;
        return new_goal;
    }
}

double LatticeReductionGoal::get_quality() {
    return quality;
}

double LatticeReductionGoal::get_max_drop() {
    double max_drop;
    if (proved) {
        max_drop = get_alpha_n(n) * n;
    } else {
        double lgn = log2(n);
        max_drop = quality * 3 * (1 + pow(3, lgn + 1) - pow(2, lgn + 2)) / 2;
        max_drop += best_slope * n;
    }
    return max_drop;
}

double LatticeReductionGoal::get_rhf() {
    double max_drop = get_max_drop();

    double slope = max_drop / n;
    double rhf = pow(2, slope / 2);
    return rhf;
}

double LatticeReductionGoal::get_slope() {
    if (proved) {
        return get_alpha_n(n);
    } else {
        return quality;
    }
}

void LatticeReductionGoal::set_best_slope(double slope) {
    assert (proved == false);
    // Need to change quality as well
    double lgn = log2(n);
    double s_guess = 3 * (1 + pow(3, lgn+1) - pow(2, lgn+2))/2;
    
    double gap = quality * s_guess / n;
    double new_gap = gap + (this->best_slope - slope);
    new_gap *= n;
    assert(new_gap > 0);
    this->quality = new_gap / s_guess;
    this->best_slope = slope;
}

LatticeReductionGoal LatticeReductionGoal::from_RHF(unsigned int n, double rhf, bool proved) {
    double slope = log2(rhf) * 2;
    return LatticeReductionGoal::from_slope(n, slope, proved);
}

LatticeReductionGoal LatticeReductionGoal::from_drop(unsigned int n, double drop, bool proved) {
    if (proved) {
        double slope = drop / (n);

        slope = std::max(slope, BKZ_BEST_SLOPE + 0.000001);

        return LatticeReductionGoal::from_slope(n, slope);
    } else {
        double slope = drop / (n - 1);
        return LatticeReductionGoal::from_slope(n, slope, proved);
    }
}

LatticeReductionGoal LatticeReductionGoal::from_slope(unsigned int n, double slope, bool proved) {
    if (proved) {
        return LatticeReductionGoal(n, slope);
    } else {
        double lgn = log2(n);
        double s_guess = 3 * (1 + pow(3, lgn+1) - pow(2, lgn+2))/2;
        double TOP_SLOPE = slope;
        if (TOP_SLOPE < BKZ_BEST_SLOPE) {
            TOP_SLOPE = BKZ_BEST_SLOPE;
        }
        double C_scale = (TOP_SLOPE - BKZ_BEST_SLOPE) * n / s_guess;

        LatticeReductionGoal ret(n, C_scale, false);
        return ret;
    }
}

}