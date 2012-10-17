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
#include "../../math/scalar.hpp"

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
 */
template<class V1, class V2>
CUDA_FUNC_GLOBAL void kernelRejectionResamplerAncestors(curandState* rng,
    const V1 lws, V2 as);

}

template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelRejectionResamplerAncestors(curandState* rng,
    const V1 lws, V2 as) {
  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw
  const int Q = gridDim.x*blockDim.x; // number of threads
  const int q = blockIdx.x*blockDim.x + threadIdx.x; // thread id

  int p, p1, p2;
  real a, lw1, lw2;

  RngGPU rng1;
  rng1.r = rng[q];

  for (p = q; p < P2; p += Q) {

    /* write result */
    as(p) = p1;
    p += Q;
  }

  rng[q] = rng1.r;
}

#endif
