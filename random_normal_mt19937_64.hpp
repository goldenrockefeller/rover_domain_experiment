#pragma once

#include <random>


class RandomNormal {
    std::mt19937_64 rng;
    std::normal_distribution<> normal_dist;
public:
    RandomNormal();
    double get();
};