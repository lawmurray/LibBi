/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICLOGDENSITYMATRIXVISITORGPU_CUH
#define BI_CUDA_UPDATER_DYNAMICLOGDENSITYMATRIXVISITORGPU_CUH

namespace bi {
/**
 * Matrix visitor for static log-density updates, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class T1, class PX, class OX>
class DynamicLogDensityMatrixVisitorGPU {
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
 * Base case of DynamicLogDensityMatrixVisitorGPU.
 */
template<class B, class T1, class PX, class OX>
class DynamicLogDensityMatrixVisitorGPU<B,empty_typelist,T1,PX,OX> {
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

template<class B, class S, class T1, class PX, class OX>
template<class T2>
inline void bi::DynamicLogDensityMatrixVisitorGPU<B,S,T1,PX,OX>::accept(
    const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const int p, const int i,
    const PX& pax, OX& x, T2& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  if (i == 0) {
    front::logDensities(t1, t2, s, p, pax, x, lp);
  } else {
    DynamicLogDensityMatrixVisitorGPU<B,pop_front,T1,PX,OX>::accept(t1, t2, s, p,
        i - 1, pax, x, lp);
  }
}

#endif
