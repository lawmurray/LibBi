/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2627 $
 * $Date: 2012-05-30 16:55:03 +0800 (Wed, 30 May 2012) $
 */
#ifndef BI_CUDA_RANDOM_RANDOM_CUH
#define BI_CUDA_RANDOM_RANDOM_CUH

inline bi::Rng<bi::ON_DEVICE>& bi::Random::getDevRng() {
  return devRngs[blockIdx.x*blockDim.x + threadIdx.x];
}

inline void bi::Random::setDevRng(const Rng<ON_DEVICE>& rng) {
  devRngs[blockIdx.x*blockDim.x + threadIdx.x] = rng;
}

#endif
