/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Random.hpp"

#ifdef ENABLE_GPU
#include "../cuda/device.hpp"
#endif

using namespace bi;

Random::Random() : own(true) {
  hostRngs = new Rng<ON_HOST>[bi_omp_max_threads];
  #ifdef ENABLE_GPU
  CUDA_CHECKED_CALL(cudaMalloc((void**)&devRngs,
      deviceIdealThreads()*sizeof(Rng<ON_DEVICE>)));
  #endif
}

Random::Random(const unsigned seed) : own(true) {
  hostRngs = new Rng<ON_HOST>[bi_omp_max_threads];
  #ifdef ENABLE_GPU
  CUDA_CHECKED_CALL(cudaMalloc((void**)&devRngs,
      deviceIdealThreads()*sizeof(Rng<ON_DEVICE>)));
  #endif

  this->seeds(seed);
}

Random::Random(const Random& o) {
  hostRngs = o.hostRngs;
  #ifdef ENABLE_GPU
  devRngs = o.devRngs;
  #endif
  own = false;
}

Random::~Random() {
  if (own) {
    delete[] hostRngs;
    hostRngs = NULL;
    #ifdef ENABLE_GPU
    CUDA_CHECKED_CALL(cudaFree(devRngs));
    devRngs = NULL;
    #endif
  }
}

void Random::seeds(const unsigned seed) {
  hostSeeds(seed);
  #ifdef ENABLE_GPU
  devSeeds(seed);
  #endif
}

void Random::hostSeeds(const unsigned seed) {
  #pragma omp parallel
  {
    this->seed(seed + bi_omp_tid);
  }
}
