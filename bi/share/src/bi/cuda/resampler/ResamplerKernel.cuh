/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_RESAMPLERKERNEL_CUH
#define BI_CUDA_RESAMPLER_RESAMPLERKERNEL_CUH

#include "misc.hpp"
#include "../cuda.hpp"

namespace bi {
/**
 * ResamplerGPU::ancestorsToOffspring() kernel.
 *
 * @tparam V1 Integer vector type.
 * @tparam V2 Integer vector type.
 *
 * @param as Ancestry.
 * @param[out] os Offspring. All elements should be initialised to 0.
 */
template<class V1, class V2>
CUDA_FUNC_GLOBAL void kernelAncestorsToOffspring(const V1 as, V2 os);

/**
 * ResamplerGPU::cumulativeOffspringToAncestorsPermute() kernel.
 *
 * @tparam V1 Integer vector type.
 * @tparam V2 Integer vector type.
 * @tparam V3 Integer vector type.
 * @tparam PrePermute Do pre-permute step?
 *
 * @param Os Cumulative offspring.
 * @param[out] as Ancestry.
 * @param[out] is Claims.
 * @param doPrePermute Either ENABLE_PRE_PERMUTE or DISABLE_PRE_PERMUTE to
 * enable or disable pre-permute step, respectively.
 */
template<class V1, class V2, class V3, class PrePermute>
CUDA_FUNC_GLOBAL void kernelCumulativeOffspringToAncestors(
    const V1 Os, V2 as, V3 is, const PrePermute doPrePermute);

/**
 * ResamplerGPU::prePermute() kernel.
 *
 * @tparam V1 Integer vector type.
 * @tparam V2 Integer vector type.
 *
 * @param as Ancestry to permute.
 * @param is[out] Claims.
 */
template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelResamplerPrePermute(const V1 as, V2 is);

/**
 * ResamplerGPU::postPermute() kernel.
 *
 * @tparam V1 Integer vector type.
 * @tparam V2 Integer vector type.
 * @tparam V3 Integer vector type.
 *
 * @param as Ancestry to permute.
 * @param is[in,out] Workspace vector, in state as returned from
 * #kernelResamplerPrePermute.
 * @param out[out] Permuted ancestry vector.
 *
 * Before calling this kernel, #kernelResamplerPrePermute should be called.
 * The remaining places in @p is are now claimed, and each thread @c i sets
 * <tt>out(is(i)) = as(i)</tt>.
 */
template<class V1, class V2, class V3>
CUDA_FUNC_GLOBAL void kernelResamplerPostPermute(const V1 as, const V2 is,
    V3 out);

}

template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelAncestorsToOffspring(const V1 as, V2 os) {
  const int P = as.size();
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  if (p < P) {
    atomicAdd(&os(as(p)), 1);
  }
}

template<class V1, class V2, class V3, class PrePermute>
CUDA_FUNC_GLOBAL void bi::kernelCumulativeOffspringToAncestors(const V1 Os,
    V2 as, V3 is, const PrePermute doPrePermute) {
  const int P = Os.size(); // number of trajectories
  const int p = blockIdx.x*blockDim.x + threadIdx.x; // trajectory id

  if (p < P) {
    int O1 = (p > 0) ? Os(p - 1) : 0;
    int O2 = Os(p);
    int o = O2 - O1;

    for (int i = 0; i < o; ++i) {
      as(O1 + i) = p;
    }
    if (doPrePermute) {
      is(p) = (o > 0) ? O1 : P;
    }
  }

//  this implementation slightly slower...
//
//  const int P = as.size(); // number of trajectories
//  const int p = blockIdx.x*blockDim.x + threadIdx.x; // trajectory id
//
//  int start = 0, end = P, O, pivot;
//  while (start != end) {
//    pivot = (start + end)/2;
//    O = Os(pivot);
//    if (p < O) {
//      end = pivot;
//    } else {
//      start = pivot + 1;
//    }
//  }
//
//  if (p < P) {
//    as(p) = start; // write will always be coalesced
//    if (doPrePermute) {
//      atomicMin(&is(start), p);
//    }
//  }
}

template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelResamplerPrePermute(const V1 as, V2 is) {
  const int P = as.size();
  const int p = blockIdx.x*blockDim.x + threadIdx.x;

  if (p < P) {
    atomicMin(&is(as(p)), p);
  }
}

template<class V1, class V2, class V3>
CUDA_FUNC_GLOBAL void bi::kernelResamplerPostPermute(const V1 as, const V2 is,
    V3 out) {
  const int P = as.size();
  const int p = blockIdx.x*blockDim.x + threadIdx.x;

  if (p < P) {
    int a = as(p), next, i;

    next = a;
    i = is(next);
    if (i != p) {
      // claim in pre-permute kernel was unsuccessful, try own spot next
      next = p;
      i = is(next);
      while (i < P) { // and chase tail of rotation until free spot
        next = i;
        i = is(next);
      }
    }

    /* write ancestor into claimed place, note the out vector is required
     * or this would cause a race condition with the read of as(p)
     * above, so this cannot be done in-place */
    out(next) = a;
  }
}

#endif
