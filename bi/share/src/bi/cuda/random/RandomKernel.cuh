/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RANDOM_RANDOMKERNEL_CUH
#define BI_CUDA_RANDOM_RANDOMKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
class Random;

/**
 * Kernel function to seed device random number generators.
 *
 * @param rng Random number generator.
 * @param seed Seed.
 */
CUDA_FUNC_GLOBAL void kernelSeeds(Random rng, const unsigned seed);

/**
 * Kernel function to fill vector with uniform variates.
 *
 * @tparam V1 Vector type.
 *
 * @param rng Random number generator.
 * @param[out] x Vector to fill.
 * @param lower Lower bound.
 * @param upper Upper bound.
 */
template<class V1>
CUDA_FUNC_GLOBAL void kernelUniforms(Random rng, V1 x,
    const typename V1::value_type lower = 0.0,
    const typename V1::value_type upper = 0.0);

/**
 * Kernel function to fill vector with Gaussian variates.
 *
 * @tparam V1 Vector type.
 *
 * @param rng Random number generator.
 * @param[out] x Vector to fill.
 * @param mu Mean.
 * @param sigma Standard deviation.
 */
template<class V1>
CUDA_FUNC_GLOBAL void kernelGaussians(Random rng, V1 x,
    const typename V1::value_type mu = 0.0,
    const typename V1::value_type sigma = 1.0);

}

#include "../../random/Random.hpp"

template<class V1>
void bi::kernelUniforms(Random rng, V1 x, const typename V1::value_type lower,
    const typename V1::value_type upper) {
  RngGPU rng1(rng.getDevRng());  // local copy, faster

  int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  for (; p < x.size(); p += Q) {
    x(p) = rng1.uniform(lower, upper);
  }

  rng.setDevRng(rng1);
}

template<class V1>
void bi::kernelGaussians(Random rng, V1 x, const typename V1::value_type mu,
    const typename V1::value_type sigma) {
  RngGPU rng1(rng.getDevRng());  // local copy, faster

  int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  for (; p < x.size(); p += Q) {
    x(p) = rng1.gaussian(mu, sigma);
  }

  rng.setDevRng(rng1);
}

#endif
