#ifndef RANDOM_NORMAL_MT19937_64_HPP
#define RANDOM_NORMAL_MT19937_64_HPP

#include <random>


class RandomNormal {
    std::mt19937_64 rng;
    std::normal_distribution<> normal_dist;
public:
    RandomNormal();
    double get();
};

#endif
