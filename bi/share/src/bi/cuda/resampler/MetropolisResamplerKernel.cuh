/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_METROPOLISRESAMPLERKERNEL_CUH
#define BI_CUDA_RESAMPLER_METROPOLISRESAMPLERKERNEL_CUH

#include "../cuda.hpp"
#include "../../math/scalar.hpp"

namespace bi {
/**
 * Metropolis resampling kernel.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Integral vector type.
 *
 * @param lws Log-weights.
 * @param as[out] Ancestry.
 * @param C Number of Metropolis steps to take.
 */
template<class V1, class V2>
CUDA_FUNC_GLOBAL void kernelMetropolisResamplerAncestors(curandState* rng, const V1 lws,
    V2 as, const int C);

}

template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelMetropolisResamplerAncestors(curandState* rng,
    const V1 lws, V2 as, const int C) {
  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw
  const int Q = gridDim.x*blockDim.x; // number of threads
  const int q = blockIdx.x*blockDim.x + threadIdx.x; // thread id

  int c, p, p1, p2;
  real a, lw1, lw2;

  RngGPU rng1;
  rng1.r = rng[q];

  for (p = q; p < P2; p += Q) {
    p1 = p;
    lw1 = lws(p1);
    for (c = 0; c < C; ++c) {
      p2 = rng1.uniformInt(0, P1 - 1);
      lw2 = lws(p2);
      if (bi::log(rng1.uniform(BI_REAL(0.0), BI_REAL(1.0))) < lw2 - lw1) {
        p1 = p2;
        lw1 = lw2;
      }
    }

    /* write result */
    as(p) = p1;
    p += Q;
  }

  rng[q] = rng1.r;
}

#endif
