/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_STRATIFIEDRESAMPLERKERNEL_CUH
#define BI_CUDA_RESAMPLER_STRATIFIEDRESAMPLERKERNEL_CUH

#include "misc.hpp"
#include "../cuda.hpp"

namespace bi {
/**
 * Stratified resampling kernel.
 *
 * @tparam V1 Vector type.
 * @tparam V2 Vector type.
 * @tparam V3 Integer vector type.
 *
 * @param alphas Strata offsets.
 * @param Ws Cumulative weights.
 * @param Os[out] Cumulative offspring.
 * @param n Number of offspring.
 */
template<class V1, class V2, class V3>
CUDA_FUNC_GLOBAL void kernelStratifiedResamplerOp(const V1 alphas,
    const V2 Ws, V3 Os, const int n);

}

template<class V1, class V2, class V3>
CUDA_FUNC_GLOBAL void bi::kernelStratifiedResamplerOp(const V1 alphas,
    const V2 Ws, V3 Os, const int n) {
  typedef typename V1::value_type T1;

  const int P = Ws.size(); // number of particles
  const T1 W = Ws(P - 1);

  const int Q = gridDim.x*blockDim.x; // number of threads
  const int q = blockIdx.x*blockDim.x + threadIdx.x; // thread id
  int p;

  for (p = q; p < P; p += Q) {
    T1 reach = Ws(p)/W*n;
    int k = bi::min(n - 1, static_cast<int>(reach));

    Os(p) = bi::min(n, static_cast<int>(reach + alphas(k)));
  }
}

#endif
