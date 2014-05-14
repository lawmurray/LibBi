/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_STATICUPDATERVISITORHOST_HPP
#define BI_HOST_UPDATER_STATICUPDATERVISITORHOST_HPP

namespace bi {
/**
 * Visitor for static updates.
 *
 * @tparam B Model type.
 * @tparam S Action type list.
 * @tparam PX Parents type.
 * @tparam OX Output type.
 */
template<class B, class S, class PX, class OX>
class StaticUpdaterVisitorHost {
public:
  static void accept(State<B,ON_HOST>& s, const int p, const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of StaticUpdaterVisitorHost.
 */
template<class B, class PX, class OX>
class StaticUpdaterVisitorHost<B,empty_typelist,PX,OX> {
public:
  static void accept(State<B,ON_HOST>& s, const int p, const PX& pax, OX& x) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class PX, class OX>
inline void bi::StaticUpdaterVisitorHost<B,S,PX,OX>::accept(
    State<B,ON_HOST>& s, const int p, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  int ix = 0;
  coord_type cox;
  while (ix < action_size<front>::value) {
    front::simulates(s, p, ix, cox, pax, x);
    ++cox;
    ++ix;
  }
  StaticUpdaterVisitorHost<B,pop_front,PX,OX>::accept(s, p, pax, x);
}

#endif
