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
#ifdef ENABLE_CUDA
#include "curand_kernel.h"
#endif

namespace bi {
class Random;

/**
 * Kernel function to seed device random number generators.
 *
 * @param rng Random number generator.
 * @param seed Seed.
 */
CUDA_FUNC_GLOBAL void kernelSeeds(curandState* rng, const unsigned seed);

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
CUDA_FUNC_GLOBAL void kernelUniforms(curandState* rng, V1 x,
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
CUDA_FUNC_GLOBAL void kernelGaussians(curandState* rng, V1 x,
    const typename V1::value_type mu = 0.0,
    const typename V1::value_type sigma = 1.0);

/**
 * Kernel function to fill vector with gamma variates.
 *
 * @tparam V1 Vector type.
 *
 * @param rng Random number generator.
 * @param[out] x Vector to fill.
 * @param alpha Shape.
 * @param beta Scale.
 */
template<class V1>
CUDA_FUNC_GLOBAL void kernelGammas(curandState* rng, V1 x,
    const typename V1::value_type alpha = 1.0,
    const typename V1::value_type beta = 1.0);

}

#include "../../random/Random.hpp"
#include "RngGPU.cuh"

template<class V1>
CUDA_FUNC_GLOBAL void bi::kernelUniforms(curandState* rng, V1 x,
    const typename V1::value_type lower,
    const typename V1::value_type upper) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  RngGPU rng1;
  rng1.r = rng[q];
  for (int p = q; p < x.size(); p += Q) {
    x(p) = rng1.uniform(lower, upper);
  }
  rng[q] = rng1.r;
}

template<class V1>
CUDA_FUNC_GLOBAL void bi::kernelGaussians(curandState* rng, V1 x,
    const typename V1::value_type mu, const typename V1::value_type sigma) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  RngGPU rng1;
  rng1.r = rng[q];
  for (int p = q; p < x.size(); p += Q) {
    x(p) = rng1.gaussian(mu, sigma);
  }
  rng[q] = rng1.r;
}

template<class V1>
CUDA_FUNC_GLOBAL void bi::kernelGammas(curandState* rng, V1 x,
    const typename V1::value_type alpha, const typename V1::value_type beta) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  RngGPU rng1;
  rng1.r = rng[q];
  for (int p = q; p < x.size(); p += Q) {
    x(p) = rng1.gamma(alpha, beta);
  }
  rng[q] = rng1.r;
}

#endif
