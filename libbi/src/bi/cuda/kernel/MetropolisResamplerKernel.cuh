/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_KERNEL_METROPOLISRESAMPLERKERNEL_CUH
#define BI_CUDA_KERNEL_METROPOLISRESAMPLERKERNEL_CUH

#include "../cuda.hpp"
#include "../../math/scalar.hpp"

namespace bi {
/**
 * @internal
 *
 * Metropolis resampling kernel.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Integral vector type.
 * @tparam V3 Integral vector type.
 *
 * @param lws Vector of sample log-weights.
 * @param qs Vector of proposal indices, containing @p L*P / 32 random
 * integers.
 * @param alphas Vector of acceptance samples, containing @p L*P random
 * numbers between zero and one.
 * @param P Number of samples.
 * @param C Number of Metropolis steps to take.
 * @param seed Base seed for random number generation.
 * @param as[out] Ancestry.
 *
 * @todo Check how proposal indices are calculated on host, can the conversion
 * to the range [0..P-1] be better done on device? If it's done on CPU already
 * while device is working, then perhaps not.
 *
 * @todo Use shared memory when log-weights will all fit.
 *
 * @todo Make C template parameter if possible for loop unrolling.
 *
 * @todo Better if alphas and qs were pitched?
 */
template<class V1, class V3>
CUDA_FUNC_GLOBAL void kernelMetropolisResamplerAncestors(const V1 lws,
    const int seed, const int P, const int C, V3 as);

}

#include "thrust/random.h"

template<class V1, class V3>
void bi::kernelMetropolisResamplerAncestors(const V1 lws,
    const int seed, const int P, const int C, V3 as) {
  const int tid = blockIdx.x*blockDim.x + threadIdx.x; // global id
  int c, p1, p2;
  real a, lw1, lw2;

  /* random number generator, uses 15 compared to 28 registers (CUDA 3.1,
   * GT200) to just use float distribution and cast to int), much faster! */
  thrust::random::taus88 rng(seed + 202099*tid/*seeds[tid]*/);
  //thrust::random::minstd_rand rng(seed + 202099*tid/*seeds[tid]*/);
  //thrust::random::uniform_int_distribution<int> p(0, P - 1);
  thrust::random::uniform_real_distribution<real> alpha(REAL(0.0), REAL(1.0));

  /* sample */
  if (tid < P) {
    p1 = tid;
    lw1 =  lws[p1];
    for (c = 0; c < C; ++c) {
      p2 = (int)(P*alpha(rng));
      lw2 = lws[p2];
      a = CUDA_EXP(lw2 - lw1);
      if (alpha(rng) < a) {
        p1 = p2;
        lw1 = lw2;
      }
    }

    /* write result */
    as[tid] = p1;
  }
}

#endif
