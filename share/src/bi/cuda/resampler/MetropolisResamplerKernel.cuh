/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_METROPOLISRESAMPLERKERNEL_CUH
#define BI_CUDA_RESAMPLER_METROPOLISRESAMPLERKERNEL_CUH

#include "misc.hpp"
#include "../cuda.hpp"

namespace bi {
/**
 * Metropolis resampling kernel with optional pre-permute.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Integer vector type.
 * @tparam V3 Integer vector type.
 * @tparam PrePermute Do pre-permute step?
 *
 * @param[in,out] rng Random number generators.
 * @param lws Log-weights.
 * @param as[out] Ancestry.
 * @param B Number of steps to take.
 * @param doPrePermute Either ENABLE_PRE_PERMUTE or DISABLE_PRE_PERMUTE to
 * enable or disable pre-permute step, respectively.
 *
 * @seealso Resampler::prePermute()
 */
template<class V1, class V2, class V3, class PrePermute>
CUDA_FUNC_GLOBAL void kernelMetropolisResamplerAncestors(curandStateSA rng,
    const V1 lws, V2 as, V3 is, const int B, const PrePermute doPrePermute);

}

template<class V1, class V2, class V3, class PrePermute>
CUDA_FUNC_GLOBAL void bi::kernelMetropolisResamplerAncestors(curandStateSA rng,
    const V1 lws, V2 as, V3 is, const int B, const PrePermute doPrePermute) {
  typedef typename V1::value_type T1;

  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw
  const int Q = gridDim.x*blockDim.x; // number of threads
  const int q = blockIdx.x*blockDim.x + threadIdx.x; // thread id

  int k, p, p1, p2;
  real a, lw1, lw2;

  RngGPU rng1;
  rng.load(q, rng1.r);

  for (p = q; p < P2; p += Q) {
    p1 = p;
    lw1 = lws(p1);

    #pragma unroll 4
    for (k = 0; k < B; ++k) {
      p2 = rng1.uniformInt(0, P1 - 1);
      lw2 = lws(p2);
      if (bi::log(rng1.uniform((T1)0.0, (T1)1.0)) < lw2 - lw1) {
        p1 = p2;
        lw1 = lw2;
      }
    }

    /* ancestor */
    as(p) = p1;

    /* pre-permute */
    if (doPrePermute) {
      atomicMin(&is(p1), p);
    }
  }

  rng.store(q, rng1.r);
}

#endif
