/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RANDOM_RANDOM_CUH
#define BI_CUDA_RANDOM_RANDOM_CUH

inline bi::RngGPU& bi::Random::getDevRng() {
  return devRngs[blockIdx.x*blockDim.x + threadIdx.x];
}

inline void bi::Random::setDevRng(const RngGPU& rng) {
  devRngs[blockIdx.x*blockDim.x + threadIdx.x] = rng;
}

#endif
