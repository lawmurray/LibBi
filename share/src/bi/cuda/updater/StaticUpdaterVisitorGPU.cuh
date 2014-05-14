/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_STATICUPDATERVISITORGPU_CUH
#define BI_CUDA_UPDATER_STATICUPDATERVISITORGPU_CUH

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
class StaticUpdaterVisitorGPU {
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
 * Base case of StaticUpdaterVisitorGPU.
 */
template<class B, class PX, class OX>
class StaticUpdaterVisitorGPU<B,empty_typelist,PX,OX> {
public:
  static CUDA_FUNC_DEVICE void accept(State<B,ON_DEVICE>& s, const int p, const int i,
      const PX& pax, OX& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class PX, class OX>
inline void bi::StaticUpdaterVisitorGPU<B,S,PX,OX>::accept(State<B,ON_DEVICE>& s, const int p,
    const int i, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  const int size = action_size<front>::value;

  if (i < size) {
    coord_type cox;
    cox.setIndex(i);
    front::simulates(s, p, i, cox, pax, x);
  } else {
    StaticUpdaterVisitorGPU<B,pop_front,PX,OX>::accept(s, p, i - size, pax, x);
  }
}

#endif
