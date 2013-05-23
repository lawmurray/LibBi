/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICSAMPLERMATRIXVISITORGPU_CUH
#define BI_CUDA_UPDATER_STATICSAMPLERMATRIXVISITORGPU_CUH

#include "../random/RngGPU.cuh"

namespace bi {
/**
 * Matrix visitor for static samples, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class PX, class OX>
class StaticSamplerMatrixVisitorGPU {
public:
  /**
   * Sample.
   *
   * @param[in,out] rng Random number generator.
   * @param s State.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static CUDA_FUNC_DEVICE void accept(RngGPU& rng, State<B,ON_DEVICE>& s, const PX& pax,
      OX& x);
};

/**
 * @internal
 *
 * Base case of StaticSamplerMatrixVisitorGPU.
 */
template<class B, class PX, class OX>
class StaticSamplerMatrixVisitorGPU<B,empty_typelist,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(RngGPU& rng, State<B,ON_DEVICE>& s, const PX& pax,
      OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class PX, class OX>
inline void bi::StaticSamplerMatrixVisitorGPU<B,S,PX,OX>::accept(
    RngGPU& rng, State<B,ON_DEVICE>& s, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  const int Q = gridDim.x*blockDim.x; // number of threads
  int p = blockIdx.x*blockDim.x + threadIdx.x;

  while (p < s.size()) {
    front::samples(rng, s, p, pax, x);
    p += Q;
  }
  StaticSamplerMatrixVisitorGPU<B,pop_front,PX,OX>::accept(rng, s, pax, x);
}

#endif
