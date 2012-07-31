/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "Random.hpp"

#ifdef ENABLE_CUDA
#include "../cuda/device.hpp"
#endif

using namespace bi;

Random::Random() : own(true) {
  hostRngs = new RngHost[bi_omp_max_threads];
  #ifdef ENABLE_CUDA
  CUDA_CHECKED_CALL(cudaMalloc((void**)&devRngs,
      deviceIdealThreads()*sizeof(RngGPU)));
  #endif
}

Random::Random(const unsigned seed) : own(true) {
  hostRngs = new RngHost[bi_omp_max_threads];
  #ifdef ENABLE_CUDA
  CUDA_CHECKED_CALL(cudaMalloc((void**)&devRngs,
      deviceIdealThreads()*sizeof(RngGPU)));
  #endif

  this->seeds(seed);
}

Random::Random(const Random& o) {
  hostRngs = o.hostRngs;
  #ifdef ENABLE_CUDA
  devRngs = o.devRngs;
  #endif
  own = false;
}

Random::~Random() {
  if (own) {
    delete[] hostRngs;
    hostRngs = NULL;
    #ifdef ENABLE_CUDA
    CUDA_CHECKED_CALL(cudaFree(devRngs));
    devRngs = NULL;
    #endif
  }
}

void Random::seeds(const unsigned seed) {
  RandomHost::seeds(*this, seed);
  #ifdef ENABLE_CUDA
  RandomGPU::seeds(*this, seed);
  #endif
}
