/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICMAXLOGDENSITYMATRIXVISITORGPU_CUH
#define BI_CUDA_UPDATER_STATICMAXLOGDENSITYMATRIXVISITORGPU_CUH

namespace bi {
/**
 * Visitor for static maximum log-density updates, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class PX, class OX>
class StaticMaxLogDensityMatrixVisitorGPU {
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
 * Base case of StaticMaxLogDensityMatrixVisitorGPU.
 */
template<class B, class PX, class OX>
class StaticMaxLogDensityMatrixVisitorGPU<B,empty_typelist,PX,OX> {
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

template<class B, class S, class PX, class OX>
template<class T1>
inline void bi::StaticMaxLogDensityMatrixVisitorGPU<B,S,PX,OX>::accept(State<B,ON_DEVICE>& s, const int p,
    const int i, const PX& pax, OX& x, T1& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  if (i == 0) {
    front::maxLogDensities(s, p, pax, x, lp);
  } else {
    StaticMaxLogDensityMatrixVisitorGPU<B,pop_front,PX,OX>::accept(s, p, i - 1, pax, x, lp);
  }
}

#endif
