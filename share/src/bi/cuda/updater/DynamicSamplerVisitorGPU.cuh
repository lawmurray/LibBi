/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICSAMPLERVISITORGPU_CUH
#define BI_CUDA_UPDATER_DYNAMICSAMPLERVISITORGPU_CUH

#include "../random/RngGPU.cuh"

namespace bi {
/**
 * Visitor for static samples, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class T1, class PX, class OX>
class DynamicSamplerVisitorGPU {
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
 * Base case of DynamicSamplerVisitorGPU.
 */
template<class B, class T1, class PX, class OX>
class DynamicSamplerVisitorGPU<B,empty_typelist,T1,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(RngGPU& rng, const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const PX& pax,
      OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class T1, class PX, class OX>
inline void bi::DynamicSamplerVisitorGPU<B,S,T1,PX,OX>::accept(RngGPU& rng,
    const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  const int Q = gridDim.x * blockDim.x;  // number of threads
  const int size = action_size<front>::value;
  int p, ix;
  coord_type cox;

  for (ix = 0; ix < size; ++ix, ++cox) {
    p = blockIdx.x * blockDim.x + threadIdx.x;
    while (p < s.size()) {
      front::samples(rng, t1, t2, s, p, ix, cox, pax, x);
      p += Q;
    }
  }
  DynamicSamplerVisitorGPU<B,pop_front,T1,PX,OX>::accept(rng, t1, t2, s, pax, x);
}

#endif
