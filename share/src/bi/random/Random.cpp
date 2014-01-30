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

bi::Random::Random() : own(true) {
  hostRngs = new RngHost[bi_omp_max_threads];
}

bi::Random::Random(const unsigned seed) : own(true) {
  hostRngs = new RngHost[bi_omp_max_threads];
  this->seeds(seed);
}

bi::Random::Random(const Random& o) {
  hostRngs = o.hostRngs;
  #ifdef ENABLE_CUDA
  devRngs = o.devRngs;
  #endif
  own = false;
}

bi::Random::~Random() {
  if (own) {
    delete[] hostRngs;
  }
}

void bi::Random::seeds(const unsigned seed) {
  RandomHost::seeds(*this, seed);
  #ifdef ENABLE_CUDA
  RandomGPU::seeds(*this, seed);
  #endif
}
