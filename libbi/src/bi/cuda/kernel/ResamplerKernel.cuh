/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_KERNEL_RESAMPLERKERNEL_CUH
#define BI_CUDA_KERNEL_RESAMPLERKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * @internal
 *
 * ResamplerDeviceImpl::permute() kernel.
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
CUDA_FUNC_GLOBAL void kernelResamplerPermute(T1* __restrict__ as, T2* __restrict__ is, const int P);

/**
 * @internal
 *
 * ResamplerDeviceImpl::copy() kernel.
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
void bi::kernelResamplerPermute(T1* __restrict__ as, T2* __restrict__ is, const int P) {
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  T1 a;
  if (id < P) {
    a = as[id];
  }
  __syncthreads();
  if (id < P) {
    int claimedBy, next;

    /* first try to claim same place as ancestor index */
    claimedBy = atomicCAS(&is[a], -1, id);

    if (claimedBy < 0) {
      // claim successful, done
      id = a;
    } else {
      // claim unsuccessful, try to claim own place...
      claimedBy = id;
      do {
        next = claimedBy;
        claimedBy = atomicCAS(&is[next], -1, id);
        // ...and continue following trace until free space found
      } while (claimedBy >= 0);
      id = next;
    }

    /* write ancestor into claimed place */
    as[id] = a;
  }
}

template<class T1, class M1>
void bi::kernelResamplerCopy(const T1* __restrict__ as, M1 s) {
  const int p = blockIdx.x*blockDim.x + threadIdx.x;
  const int id = blockIdx.y*blockDim.y + threadIdx.y;

  if (p < s.size1() && id < s.size2() && as[p] != p) {
    s(p,id) = s(as[p],id);
  }
}

#endif
