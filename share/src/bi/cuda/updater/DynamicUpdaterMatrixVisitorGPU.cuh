/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICUPDATERMATRIXVISITORGPU_CUH
#define BI_CUDA_UPDATER_DYNAMICUPDATERMATRIXVISITORGPU_CUH

namespace bi {
/**
 * Matrix visitor for dynamic update, on GPU.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class T1, class PX, class OX>
class DynamicUpdaterMatrixVisitorGPU {
public:
  /**
   * Update.
   *
   * @param t1 Start of time interval.
   * @param t2 End of time interval.
   * @param s State.
   * @param p Trajectory id.
   * @param i Variable id.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static CUDA_FUNC_DEVICE void accept(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const int p,
      const int i, const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of DynamicUpdaterMatrixVisitorGPU.
 */
template<class B, class T1, class PX, class OX>
class DynamicUpdaterMatrixVisitorGPU<B,empty_typelist,T1,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const int p,
      const int i, const PX& pax, OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class T1, class PX, class OX>
inline void bi::DynamicUpdaterMatrixVisitorGPU<B,S,T1,PX,OX>::accept(
    const T1 t1, const T1 t2, State<B,ON_DEVICE>& s, const int p, const int i,
    const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  if (i == 0) {
    front::simulates(t1, t2, s, p, pax, x);
  } else {
    DynamicUpdaterMatrixVisitorGPU<B,pop_front,T1,PX,OX>::accept(t1, t2, s, p,
        i - 1, pax, x);
  }
}

#endif
