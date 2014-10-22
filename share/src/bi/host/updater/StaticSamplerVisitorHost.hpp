/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_STATICSAMPLERVISITORHOST_HPP
#define BI_HOST_UPDATER_STATICSAMPLERVISITORHOST_HPP

namespace bi {
/**
 * Visitor for StaticSamplerHost.
 */
template<class B, class S, class R1, class PX, class OX>
class StaticSamplerVisitorHost {
public:
  static void accept(R1& rng, State<B,ON_HOST>& s, const int p, const PX& pax,
      OX& x);
};

/**
 * @internal
 *
 * Base case of StaticSamplerVisitorHost.
 */
template<class B, class R1, class PX, class OX>
class StaticSamplerVisitorHost<B,empty_typelist,R1,PX,OX> {
public:
  static void accept(R1& rng, State<B,ON_HOST>& s, const int p, const PX& pax,
      OX& x) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../traits/action_traits.hpp"

template<class B, class S, class R1, class PX, class OX>
void bi::StaticSamplerVisitorHost<B,S,R1,PX,OX>::accept(R1& rng,
    State<B,ON_HOST>& s, const int p, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef typename front::coord_type coord_type;

  int ix = 0;
  coord_type cox;
  while (ix < action_size<front>::value) {
    front::samples(rng, s, p, ix, cox, pax, x);
    ++cox;
    ++ix;
  }
  StaticSamplerVisitorHost<B,pop_front,R1,PX,OX>::accept(rng, s, p, pax, x);
}

#endif
