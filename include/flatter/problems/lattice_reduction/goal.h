#pragma once

#include <flatter/data/lattice/profile.h>

#define BKZ_BEST_SLOPE 0.031281 
#define HERMITE_BEST_SLOPE 0.41503749927884365
#define DEFAULT_G 3

namespace flatter {

class LatticeReductionGoal {
public:
    LatticeReductionGoal();
    LatticeReductionGoal(
        unsigned int n,
        double quality,
        bool proved
    );
    LatticeReductionGoal(
        unsigned int n,
        double top_level_slope,
        double base_slope = BKZ_BEST_SLOPE,
        double g = DEFAULT_G,
        unsigned int top_N = 0
    );

    bool check(Profile profile);
    LatticeReductionGoal subgoal(unsigned int start, unsigned int end);

    double get_rhf();
    double get_quality();
    double get_slope();
    double get_alpha_n(unsigned int n);
    double get_max_drop();
    void set_best_slope(double best_slope);

    static LatticeReductionGoal from_RHF(unsigned int n, double rhf, bool proved = false);
    static LatticeReductionGoal from_drop(unsigned int n, double drop, bool proved = false);
    static LatticeReductionGoal from_slope(unsigned int n, double slope, bool proved = false);

private:

    unsigned int n;
    unsigned int top_N;
    double quality;
    double best_slope;
    double top_slope;
    double g;
    double log_g;

    bool proved;
};

}