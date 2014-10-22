/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICLOGDENSITYVISITORGPU_CUH
#define BI_CUDA_UPDATER_DYNAMICLOGDENSITYVISITORGPU_CUH

namespace bi {
/**
 * Visitor for static log-density updates, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class T1, class PX, class OX>
class DynamicLogDensityVisitorGPU {
public:
  /**
   * Update.
   *
   * @tparam T2 Scalar type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param i Variable id.
   * @param pax Parents.
   * @param[out] x Output.
   * @param[in,out] lp Log-density.
   */
  template<class T2>
  static CUDA_FUNC_DEVICE void accept(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x, T2& lp);
};

/**
 * @internal
 *
 * Base case of DynamicLogDensityVisitorGPU.
 */
template<class B, class T1, class PX, class OX>
class DynamicLogDensityVisitorGPU<B,empty_typelist,T1,PX,OX> {
public:
  template<class T2>
  static CUDA_FUNC_DEVICE void accept(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x, T2& lp) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class T1, class PX, class OX>
template<class T2>
inline void bi::DynamicLogDensityVisitorGPU<B,S,T1,PX,OX>::accept(const T1 t1,
    const T1 t2, State<B,ON_DEVICE>& s, const int p, const int i,
    const PX& pax, OX& x, T2& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  const int size = action_size<front>::value;

  if (i < size) {
    coord_type cox;
    cox.setIndex(i);
    front::logDensities(t1, t2, s, p, i, cox, pax, x, lp);
  } else {
    DynamicLogDensityVisitorGPU<B,pop_front,T1,PX,OX>::accept(t1, t2, s, p,
        i - size, pax, x, lp);
  }
}

#endif
