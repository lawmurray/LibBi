/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_RESAMPLERKERNEL_CUH
#define BI_CUDA_RESAMPLER_RESAMPLERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * ResamplerGPU::permute() kernel.
 *
 * @tparam T1 Integer type.
 * @tparam T2 Signed integer type.
 *
 * @param as[in,out] Ancestry to be permuted.
 * @param is[in,out] Workspace vector. All elements should be initialised to
 * -1.
 * @param P Number of particles.
 */
template<class T1, class T2>
CUDA_FUNC_GLOBAL void kernelResamplerPrePermute(T1* __restrict__ as,
    T2* __restrict__ is, const int P);

/**
 * ResamplerGPU::permute() kernel.
 *
 * @tparam T1 Integer type.
 * @tparam T2 Signed integer type.
 *
 * @param as[in,out] Ancestry to be permuted.
 * @param is[in,out] Workspace vector. All elements should be initialised to
 * -1.
 * @param P Number of particles.
 *
 * This function is only correct for @p as consisting of indices,
 * <tt>0..P-1</tt>, in all other cases it returns a valid ancestry vector,
 * but one that does not necessarily satisfy the requirement that, if @c i
 * appears at least once in the ancestry vector, then <tt>as[i] == i</tt>.
 * Running #kernelResamplerPrePermute with the same arguments before calling
 * kernelResamplerPermute provides this guarantee.
 */
template<class T1, class T2>
CUDA_FUNC_GLOBAL void kernelResamplerPermute(T1* __restrict__ as,
    T2* __restrict__ is, const int P);

/**
 * @internal
 *
 * ResamplerGPU::copy() kernel.
 *
 * @tparam T1 Integer type.
 * @tparam M1 Matrix type.
 *
 * @param as Ancestry.
 * @param[in,out] s State.
 */
template<class T1, class M1>
CUDA_FUNC_GLOBAL void kernelResamplerCopy(const T1* __restrict__ as, M1 s);

}

template<class T1, class T2>
CUDA_FUNC_GLOBAL void bi::kernelResamplerPrePermute(T1* __restrict__ as, T2* __restrict__ is,
    const int P) {
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < P) {
    T1 a = as[id];
    atomicCAS(&is[a], -1, id);
  }
}

template<class T1, class T2>
CUDA_FUNC_GLOBAL void bi::kernelResamplerPermute(T1* __restrict__ as, T2* __restrict__ is,
    const int P) {
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < P) {
    T1 a = as[id];
    int claimedBy, next;

    /* first try to claim same place as ancestor index */
    next = a;
    claimedBy = atomicCAS(&is[next], -1, id);

    if (claimedBy != id && claimedBy >= 0) {
      // claim unsuccessful, try to claim own place...
      claimedBy = id;
      do {
        next = claimedBy;
        claimedBy = atomicCAS(&is[next], -1, id);
        // ...and continue following trace until free space found
      } while (claimedBy != id && claimedBy >= 0);
    }

    /* write ancestor into claimed place */
    as[next] = a;
  }
}

template<class T1, class M1>
CUDA_FUNC_GLOBAL void bi::kernelResamplerCopy(const T1* __restrict__ as, M1 s) {
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = blockIdx.y*blockDim.y + threadIdx.y;

  if (p < s.size1() && id < s.size2() && as[p] != p) {
    s(p,id) = s(as[p],id);
  }
}

#endif
