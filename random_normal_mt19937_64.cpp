# include "random_normal_mt19937_64.hpp"


RandomNormal::RandomNormal() {
    // Make a random number engine
    rng = std::mt19937_64(std::random_device()());

    normal_dist = std::normal_distribution<>(0., 1.);
    
}

double RandomNormal::get() {
    return normal_dist(rng);
    //return 0.;
}

