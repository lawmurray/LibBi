/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DYNAMICUPDATERVISITORGPU_CUH
#define BI_CUDA_UPDATER_DYNAMICUPDATERVISITORGPU_CUH

namespace bi {
/**
 * Visitor for dynamic update, on GPU.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class T1, class PX, class OX>
class DynamicUpdaterVisitorGPU {
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
  static CUDA_FUNC_DEVICE void accept(const T1 t1,
      const T1 t2, State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of DynamicUpdaterVisitorGPU.
 */
template<class B, class T1, class PX, class OX>
class DynamicUpdaterVisitorGPU<B,empty_typelist,T1,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(const T1 t1,
      const T1 t2, State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class T1, class PX, class OX>
inline void bi::DynamicUpdaterVisitorGPU<B,S,T1,PX,OX>::accept(const T1 t1,
    const T1 t2, State<B,ON_DEVICE>& s, const int p, const int i,
    const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  const int size = action_size<front>::value;

  if (i < size) {
    coord_type cox(i);
    front::simulates(t1, t2, s, p, i, cox, pax, x);
  } else {
    DynamicUpdaterVisitorGPU<B,pop_front,T1,PX,OX>::accept(t1, t2, s, p,
        i - size, pax, x);
  }
}

#endif
