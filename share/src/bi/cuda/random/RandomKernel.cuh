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
#include "curandStateSA.hpp"
#ifdef ENABLE_CUDA
#include "curand_kernel.h"
#endif

namespace bi {
class Random;

/**
 * Kernel function to seed device random number generators.
 *
 * @param[in,out] rng Random number generator.
 * @param seed Seed.
 */
CUDA_FUNC_GLOBAL void kernelSeeds(curandStateSA rng, const unsigned seed);

/**
 * Kernel function to fill vector with uniform variates.
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] rng Random number generator.
 * @param[out] x Vector to fill.
 * @param lower Lower bound.
 * @param upper Upper bound.
 */
template<class V1>
CUDA_FUNC_GLOBAL void kernelUniforms(curandStateSA rng, V1 x,
    const typename V1::value_type lower = 0.0,
    const typename V1::value_type upper = 0.0);

/**
 * Kernel function to fill vector with Gaussian variates.
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] rng Random number generator.
 * @param[out] x Vector to fill.
 * @param mu Mean.
 * @param sigma Standard deviation.
 */
template<class V1>
CUDA_FUNC_GLOBAL void kernelGaussians(curandStateSA rng, V1 x,
    const typename V1::value_type mu = 0.0,
    const typename V1::value_type sigma = 1.0);

/**
 * Kernel function to fill vector with gamma variates.
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] rng Random number generator.
 * @param[out] x Vector to fill.
 * @param alpha Shape.
 * @param beta Scale.
 */
template<class V1>
CUDA_FUNC_GLOBAL void kernelGammas(curandStateSA rng, V1 x,
    const typename V1::value_type alpha = 1.0,
    const typename V1::value_type beta = 1.0);

/**
 * Kernel function to fill vector with poisson variates.
 *
 * @tparam V1 Vector type.
 *
 * @param[in,out] rng Random number generator.
 * @param[out] x Vector to fill.
 * @param lambda Rate.
 */
template<class V1>
CUDA_FUNC_GLOBAL void bi::kernelPoissons(curandStateSA rng, V1 x,
    const typename V1::value_type lambda); 

/**
 * Kernel function to fill vector with binomial variates.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 *
 * @param[in,out] rng Random number generator.
 * @param[out] x Vector to fill.
 * @param n Size.
 * @param p Probability.
 */
template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelBinomials(curandStateSA rng, V1 x,
    const typename V1::value_type n, const typename V2::value_type p);
}

#include "../../random/Random.hpp"
#include "RngGPU.cuh"

template<class V1>
CUDA_FUNC_GLOBAL void bi::kernelUniforms(curandStateSA rng, V1 x,
    const typename V1::value_type lower,
    const typename V1::value_type upper) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  RngGPU rng1;
  rng.load(q, rng1.r);
  for (int p = q; p < x.size(); p += Q) {
    x(p) = rng1.uniform(lower, upper);
  }
  rng.store(q, rng1.r);
}

template<class V1>
CUDA_FUNC_GLOBAL void bi::kernelGaussians(curandStateSA rng, V1 x,
    const typename V1::value_type mu, const typename V1::value_type sigma) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  RngGPU rng1;
  rng.load(q, rng1.r);
  for (int p = q; p < x.size(); p += Q) {
    x(p) = rng1.gaussian(mu, sigma);
  }
  rng.store(q, rng1.r);
}

template<class V1>
CUDA_FUNC_GLOBAL void bi::kernelGammas(curandStateSA rng, V1 x,
    const typename V1::value_type alpha, const typename V1::value_type beta) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  RngGPU rng1;
  rng.load(q, rng1.r);
  for (int p = q; p < x.size(); p += Q) {
    x(p) = rng1.gamma(alpha, beta);
  }
  rng.store(q, rng1.r);
}

template<class V1>
CUDA_FUNC_GLOBAL void bi::kernelPoissons(curandStateSA rng, V1 x,
    const typename V1::value_type lambda) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  RngGPU rng1;
  rng.load(q, rng1.r);
  for (int p = q; p < x.size(); p += Q) {
    x(p) = rng1.poisson(lambda);
  }
  rng.store(q, rng1.r);
}

template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelBinomials(curandStateSA rng, V1 x,
    const typename V1::value_type n, const typename V2::value_type p) {
  const int q = blockIdx.x*blockDim.x + threadIdx.x;
  const int Q = blockDim.x*gridDim.x;

  RngGPU rng1;
  rng.load(q, rng1.r);
  for (int p = q; p < x.size(); p += Q) {
    x(p) = rng1.binomial(n, p);
  }
  rng.store(q, rng1.r);
}

#endif
