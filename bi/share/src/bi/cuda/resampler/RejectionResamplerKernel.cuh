/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_REJECTIONRESAMPLERKERNEL_CUH
#define BI_CUDA_RESAMPLER_REJECTIONRESAMPLERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Rejection resampling kernel.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Integral vector type.
 *
 * @param rng Random number generators.
 * @param lws Log-weights.
 * @param as[out] Ancestry.
 * @param maxLogWeight Maximum log-weight.
 */
template<class V1, class V2>
CUDA_FUNC_GLOBAL void kernelRejectionResamplerAncestors(curandState* rng,
    const V1 lws, V2 as, const typename V1::value_type maxLogWeight);

}

#include "../shared.cuh"

template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelRejectionResamplerAncestors(curandState* rng,
    const V1 lws, V2 as, const typename V1::value_type maxLogWeight) {
  typedef typename V1::value_type T1;

  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw
  const int Q = gridDim.x*blockDim.x; // number of threads
  const int q = blockIdx.x*blockDim.x + threadIdx.x; // thread id

  int p, p2;
  real lalpha, lw2;

  RngGPU rng1;
  rng1.r = rng[q];

  for (p = q; p < P2; p += Q) {
    /* first proposal */
    if (p < P2/P1*P1) {
      /* death jump (stratified uniform) proposal */
      p2 = p % P1;
    } else {
      /* random proposal */
      p2 = rng1.uniformInt(0, P1 - 1);
    }
    lw2 = lws(p2) - maxLogWeight;
    lalpha = bi::log(rng1.uniform((T1)0.0, (T1)1.0));

    /* rejection loop */
    while (lalpha > lw2) {
      p2 = rng1.uniformInt(0, P1 - 1);
      lw2 = lws(p2) - maxLogWeight;
      lalpha = bi::log(rng1.uniform((T1)0.0, (T1)1.0));
    }

    /* write result */
    as(p) = p2;
  }

  // less warp divergence, but no death-jump, and writes aren't coalesced, so
  // slower overall
  //
  //p = q;
  //do {
  //  p2 = rng1.uniformInt(0, P1 - 1);
  //  lw2 = lws(p2) - maxLogWeight;
  //  lalpha = bi::log(rng1.uniform((T1)0.0, (T1)1.0));
  //  if (lalpha < lw2) {
  //    as(p) = p2;
  //    p += Q;
  //  }
  //} while (p < P2);

  rng[q] = rng1.r;
}

#endif
