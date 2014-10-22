/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_DYNAMICMAXLOGDENSITYVISITORHOST_HPP
#define BI_HOST_UPDATER_DYNAMICMAXLOGDENSITYVISITORHOST_HPP

namespace bi {
/**
 * Visitor for DynamicMaxLogDensityHost.
 */
template<class B, class S, class PX, class OX>
class DynamicMaxLogDensityVisitorHost {
public:
  template<class T1, class T2>
  static void accept(const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      const int p, const PX& pax, OX& x, T2& lp);
};

/**
 * @internal
 *
 * Base case of DynamicMaxLogDensityVisitorHost.
 */
template<class B, class PX, class OX>
class DynamicMaxLogDensityVisitorHost<B,empty_typelist,PX,OX> {
public:
  template<class T1, class T2>
  static void accept(const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      const int p, const PX& pax, OX& x, T2& lp) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class PX, class OX>
template<class T1, class T2>
void bi::DynamicMaxLogDensityVisitorHost<B,S,PX,OX>::accept(const T1 t1,
    const T1 t2, State<B,ON_HOST>& s, const int p, const PX& pax, OX& x,
    T2& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  int ix = 0;
  coord_type cox;
  while (ix < action_size<front>::value) {
    front::maxLogDensities(t1, t2, s, p, ix, cox, pax, x, lp);
    ++cox;
    ++ix;
  }
  DynamicMaxLogDensityVisitorHost<B,pop_front,PX,OX>::accept(t1, t2, s, p,
      pax, x, lp);
}

#endif
