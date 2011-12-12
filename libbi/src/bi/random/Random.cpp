/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/Random.cpp
 */
#include "Random.hpp"

using namespace bi;

Random::Random() : rng(bi_omp_max_threads) {
  #ifndef USE_CPU
  CURAND_CHECKED_CALL(curandCreateGenerator(&devRng, CURAND_RNG_PSEUDO_DEFAULT));
  #endif
}

Random::Random(const unsigned seed) : rng(bi_omp_max_threads) {
  #ifndef USE_CPU
  CURAND_CHECKED_CALL(curandCreateGenerator(&devRng, CURAND_RNG_PSEUDO_DEFAULT));
  #endif
  this->seed(seed);
}

Random::~Random() {
  #ifndef USE_CPU
  CURAND_CHECKED_CALL(curandDestroyGenerator(devRng));
  #endif
}

void Random::seed(const unsigned seed) {
  originalSeed = seed;

  /* seed host generators */
  unsigned i;
  for (i = 0; i < rng.size(); ++i) {
    rng[i].seed(seed + i);
  }

  /* seed device generator */
  #ifndef USE_CPU
  CURAND_CHECKED_CALL(curandSetPseudoRandomGeneratorSeed(devRng, seed));
  #endif
}

void Random::reset() {
  #ifndef USE_CPU
  /* just re-seeding seems insufficient to reproduce same sequence, so destroy
   * and re-create */
  CURAND_CHECKED_CALL(curandDestroyGenerator(devRng));
  CURAND_CHECKED_CALL(curandCreateGenerator(&devRng, CURAND_RNG_PSEUDO_DEFAULT));
  #endif

  this->seed(originalSeed);
}
