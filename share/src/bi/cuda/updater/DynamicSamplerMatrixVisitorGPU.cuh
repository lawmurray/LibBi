/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICSAMPLERMATRIXVISITORGPU_CUH
#define BI_CUDA_UPDATER_DYNAMICSAMPLERMATRIXVISITORGPU_CUH

#include "../random/RngGPU.cuh"

namespace bi {
/**
 * Matrix visitor for static samples, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class T1, class PX, class OX>
class DynamicSamplerMatrixVisitorGPU {
public:
  /**
   * Sample.
   *
   * @param[in,out] rng Random number generator.
   * @param t1 Start of time interval.
   * @param t2 End of time interval.
   * @param s State.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static CUDA_FUNC_DEVICE void accept(RngGPU& rng, const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const PX& pax,
      OX& x);
};

/**
 * @internal
 *
 * Base case of DynamicSamplerMatrixVisitorGPU.
 */
template<class B, class T1, class PX, class OX>
class DynamicSamplerMatrixVisitorGPU<B,empty_typelist,T1,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(RngGPU& rng, const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const PX& pax,
      OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class T1, class PX, class OX>
inline void bi::DynamicSamplerMatrixVisitorGPU<B,S,T1,PX,OX>::accept(RngGPU& rng,
    const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  const int Q = gridDim.x * blockDim.x;  // number of threads
  int p = blockIdx.x * blockDim.x + threadIdx.x;

  while (p < s.size()) {
    front::samples(rng, t1, t2, s, p, pax, x);
    p += Q;
  }
  DynamicSamplerMatrixVisitorGPU<B,pop_front,T1,PX,OX>::accept(rng, t1, t2, s, pax, x);
}

#endif
