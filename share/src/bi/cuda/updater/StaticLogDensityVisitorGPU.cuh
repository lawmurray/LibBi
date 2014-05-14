/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICLOGDENSITYVISITORGPU_CUH
#define BI_CUDA_UPDATER_STATICLOGDENSITYVISITORGPU_CUH

namespace bi {
/**
 * Visitor for static log-density updates, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class PX, class OX>
class StaticLogDensityVisitorGPU {
public:
  /**
   * Update.
   *
   * @tparam T1 Scalar type.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param i Variable id.
   * @param pax Parents.
   * @param[out] x Output.
   * @param[in,out] lp Log-density.
   */
  template<class T1>
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x, T1& lp);
};

/**
 * @internal
 *
 * Base case of StaticLogDensityVisitorGPU.
 */
template<class B, class PX, class OX>
class StaticLogDensityVisitorGPU<B,empty_typelist,PX,OX> {
public:
  template<class T1>
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x, T1& lp) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class PX, class OX>
template<class T1>
inline void bi::StaticLogDensityVisitorGPU<B,S,PX,OX>::accept(State<B,ON_DEVICE>& s, const int p,
    const int i, const PX& pax, OX& x, T1& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  const int size = action_size<front>::value;

  if (i < size) {
    coord_type cox;
    cox.setIndex(i);
    front::logDensities(s, p, i, cox, pax, x, lp);
  } else {
    StaticLogDensityVisitorGPU<B,pop_front,PX,OX>::accept(s, p, i - size, pax, x, lp);
  }
}

#endif
