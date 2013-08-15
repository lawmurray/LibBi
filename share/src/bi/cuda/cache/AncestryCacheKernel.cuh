/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_CACHE_ANCESTRYCACHEKERNEL_CUH
#define BI_CUDA_CACHE_ANCESTRYCACHEKERNEL_CUH

#include "../cuda.hpp"

namespace bi {
/**
 * Kernel function for ancestry tree pruning.
 *
 * @tparam V1 Integer vector type.
 * @tparam V2 Integer vector type.
 *
 * @param as Ancestors.
 * @param os Offspring.
 * @param ls Leaves.
 * @param[out] numRemoved Number of nodes removed by each thread.
 */
template<class V1, class V2>
CUDA_FUNC_GLOBAL void kernelAncestryCachePrune(V1 as, V1 os, V1 ls, V2 numRemoved);

}

template<class V1, class V2>
CUDA_FUNC_GLOBAL void bi::kernelAncestryCachePrune(V1 as, V1 os, V1 ls, V2 numRemoved) {
  const int N = ls.size();
  const int i = blockIdx.x*blockDim.x + threadIdx.x;
  int j, o, m = 0;

  if (i < N) {
    j = ls(i);
    o = os(j);
    while (o == 0) {
      ++m;
      j = as(j);
      if (j >= 0) {
        o = atomicSub(&os(j), 1) - 1;
      } else {
        break;
      }
    }
    numRemoved(i) = m;
  }
}

#endif
