/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_REJECTIONRESAMPLERKERNEL_CUH
#define BI_CUDA_RESAMPLER_REJECTIONRESAMPLERKERNEL_CUH

#include "misc.hpp"
#include "../cuda.hpp"

namespace bi {
/**
 * Rejection resampling kernel with optional pre-permute.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Integer vector type.
 * @tparam V3 Integer vector type.
 * @tparam PrePermute Do pre-permute step?
 *
 * @param[in,out] rng Random number generators.
 * @param lws Log-weights.
 * @param as[out] Ancestry.
 * @param is[in,out] Workspace vector. All elements should be initialised to
 * <tt>as.size()</tt>.
 * @param maxLogWeight Maximum log-weight.
 * @param doPrePermute Either ENABLE_PRE_PERMUTE or DISABLE_PRE_PERMUTE to
 * enable or disable pre-permute step, respectively.
 *
 * @seealso Resampler::prePermute()
 */
template<class V1, class V2, class V3, class PrePermute>
CUDA_FUNC_GLOBAL void kernelRejectionResamplerAncestors(curandStateSA rng,
    const V1 lws, V2 as, V3 is, const typename V1::value_type maxLogWeight,
    const PrePermute doPrePermute);

/**
 * Rejection resampling kernel with optional pre-permute and variable
 * task-length handling. This achieves better performance than
 * kernelRejectionResamplerAncestorsPrePermute() for very large numbers of
 * particles.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Integer vector type.
 * @tparam V3 Integer vector type.
 * @tparam PrePermute Do pre-permute step?
 *
 * @param[in,out] rng Random number generators.
 * @param lws Log-weights.
 * @param as[out] Ancestry.
 * @param is[in,out] Workspace vector. All elements should be initialised to
 * <tt>as.size()</tt>.
 * @param maxLogWeight Maximum log-weight.
 * @param doPrePermute Either ENABLE_PRE_PERMUTE or DISABLE_PRE_PERMUTE to
 * enable or disable pre-permute step, respectively.
 *
 * @seealso Resampler::prePermute()
 */
template<class V1, class V2, class V3, class PrePermute>
CUDA_FUNC_GLOBAL void kernelRejectionResamplerAncestors2(curandStateSA rng,
    const V1 lws, V2 as, V3 is, const typename V1::value_type maxLogWeight,
    const PrePermute doPrePermute);

}

#include "../shared.cuh"

template<class V1, class V2, class V3, class PrePermute>
CUDA_FUNC_GLOBAL void bi::kernelRejectionResamplerAncestors(
    curandStateSA rng, const V1 lws, V2 as, V3 is,
    const typename V1::value_type maxLogWeight,
    const PrePermute doPrePermute) {
  typedef typename V1::value_type T1;

  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw
  const int Q = gridDim.x*blockDim.x; // number of threads
  const int q = blockIdx.x*blockDim.x + threadIdx.x; // thread id

  int p, p2, i;
  real lalpha, lw2;
  bool accept;

  RngGPU rng1;
  rng.load(q, rng1.r);

  for (p = q; p < P2; p += Q) {
    /* first proposal */
    p2 = (p < P2/P1*P1) ? p % P1 : rng1.uniformInt(0, P1 - 1);
    lw2 = lws(p2) - maxLogWeight;
    lalpha = bi::log(rng1.uniform((T1)0.0, (T1)1.0));

    /* rejection loop */
    while (lalpha > lw2) {
      p2 = rng1.uniformInt(0, P1 - 1);
      lw2 = lws(p2) - maxLogWeight;
      lalpha = bi::log(rng1.uniform((T1)0.0, (T1)1.0));
    }

    /* ancestor */
    as(p) = p2;

    /* pre-permute */
    if (doPrePermute) {
      atomicMin(&is(p2), p);
    }
  }

  rng.store(q, rng1.r);
}

template<class V1, class V2, class V3, class PrePermute>
CUDA_FUNC_GLOBAL void bi::kernelRejectionResamplerAncestors2(
    curandStateSA rng, const V1 lws, V2 as, V3 is,
    const typename V1::value_type maxLogWeight,
    const PrePermute doPrePermute) {
  typedef typename V1::value_type T1;

  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw
  const int Q = gridDim.x*blockDim.x; // number of threads
  const int q = blockIdx.x*blockDim.x + threadIdx.x; // thread id
  const int bufSize = 4;

  int p, p2, i;
  int* __restrict__ buf = reinterpret_cast<int*>(shared_mem);
  real lalpha, lw2;
  bool accept;

  RngGPU rng1;
  rng.load(q, rng1.r);

  for (p = q; p < P2; p += bufSize*Q) {
    /* rejection sample to fill buffer */
    i = 0;
    accept = true;
    do {
      p2 = (accept && p < P2/P1*P1) ? (p + i*Q) % P1 : rng1.uniformInt(0, P1 - 1);
      lw2 = lws(p2) - maxLogWeight;
      lalpha = bi::log(rng1.uniform((T1)0.0, (T1)1.0));
      accept = lalpha <= lw2;
      if (accept) {
        buf[blockDim.x*i + threadIdx.x] = p2;
        ++i;
      }
    } while (i < bufSize && p + i*Q < P2);

    /* write buffer (coalesced) */
    for (i = 0; i < bufSize && p + i*Q < P2; ++i) {
      p2 = buf[blockDim.x*i + threadIdx.x];

      /* ancestor */
      as(p + i*Q) = p2;

      /* pre-permute */
      if (doPrePermute) {
        atomicMin(&is(p2), p + i*Q);
      }
    }
  }

  rng.store(q, rng1.r);
}

#endif
