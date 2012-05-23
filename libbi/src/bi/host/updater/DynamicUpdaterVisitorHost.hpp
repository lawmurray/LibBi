/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_DYNAMICUPDATERVISITORHOST_HPP
#define BI_HOST_UPDATER_DYNAMICUPDATERVISITORHOST_HPP

namespace bi {
/**
 * Visitor for dynamic updates.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class T1, class PX, class OX>
class DynamicUpdaterVisitorHost {
public:
  /**
   * Update d-net.
   *
   * @param t1 Start of time interval.
   * @param t2 End of time interval.
   * @param p Trajectory id.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static void accept(const T1 t1, const T1 t2, const int p, const PX& pax,
      OX& x);
};

/**
 * @internal
 *
 * Base case of DynamicUpdaterVisitorHost.
 */
template<class B, class T1, class PX, class OX>
class DynamicUpdaterVisitorHost<B,empty_typelist,T1,PX,OX> {
public:
  static void accept(const T1 t1, const T1 t2, const int p, const PX& pax,
      OX& x) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/target_traits.hpp"

template<class B, class S, class T1, class PX, class OX>
inline void bi::DynamicUpdaterVisitorHost<B,S,T1,PX,OX>::accept(
    const T1 t1, const T1 t2, const int p, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename target_type::coord_type coord_type;

  int ix = 0;
  coord_type cox;
  while (ix < target_size<target_type>::value) {
    front::f(t1, t2, p, ix, cox, pax, x);
    ++cox;
    ++ix;
  }
  DynamicUpdaterVisitorHost<B,pop_front,T1,PX,OX>::accept(t1, t2, p, pax, x);
}

#endif
