/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_STATICLOGDENSITYVISITORHOST_HPP
#define BI_HOST_UPDATER_STATICLOGDENSITYVISITORHOST_HPP

namespace bi {
/**
 * Visitor for StaticLogDensityHost.
 */
template<class B, class S, class PX, class OX>
class StaticLogDensityVisitorHost {
public:
  template<class T1>
  static void accept(State<B,ON_HOST>& s, const int p, const PX& pax, OX& x,
      T1& lp);
};

/**
 * @internal
 *
 * Base case of StaticLogDensityVisitorHost.
 */
template<class B, class PX, class OX>
class StaticLogDensityVisitorHost<B,empty_typelist,PX,OX> {
public:
  template<class T1>
  static void accept(State<B,ON_HOST>& s, const int p, const PX& pax, OX& x,
      T1& lp) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class PX, class OX>
template<class T1>
void bi::StaticLogDensityVisitorHost<B,S,PX,OX>::accept(State<B,ON_HOST>& s,
    const int p, const PX& pax, OX& x, T1& lp) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  int ix = 0;
  coord_type cox;
  while (ix < action_size<front>::value) {
    front::logDensities(s, p, ix, cox, pax, x, lp);
    ++cox;
    ++ix;
  }
  StaticLogDensityVisitorHost<B,pop_front,PX,OX>::accept(s, p, pax, x, lp);
}

#endif
