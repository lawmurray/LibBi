/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICUPDATERMATRIXVISITORGPU_CUH
#define BI_CUDA_UPDATER_STATICUPDATERMATRIXVISITORGPU_CUH

namespace bi {
/**
 * Visitor for static updates, on device.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class PX, class OX>
class StaticUpdaterMatrixVisitorGPU {
public:
  /**
   * Update.
   *
   * @param p Trajectory id.
   * @param i Variable id.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of StaticUpdaterMatrixVisitorGPU.
 */
template<class B, class PX, class OX>
class StaticUpdaterMatrixVisitorGPU<B,empty_typelist,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class PX, class OX>
inline void bi::StaticUpdaterMatrixVisitorGPU<B,S,PX,OX>::accept(State<B,ON_DEVICE>& s, const int p,
    const int i, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  if (i == 0) {
    front::simulates(s, p, pax, x);
  } else {
    StaticUpdaterMatrixVisitorGPU<B,pop_front,PX,OX>::accept(s, p, i - 1, pax, x);
  }
}

#endif
