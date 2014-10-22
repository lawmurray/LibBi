/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_STATICSAMPLERMATRIXVISITORHOST_HPP
#define BI_HOST_UPDATER_STATICSAMPLERMATRIXVISITORHOST_HPP

namespace bi {
/**
 * Visitor for StaticSamplerHost.
 */
template<class B, class S, class R1, class PX, class OX>
class StaticSamplerMatrixVisitorHost {
public:
  static void accept(R1& rng, State<B,ON_HOST>& s, const int p, const PX& pax,
      OX& x);
};

/**
 * @internal
 *
 * Base case of StaticSamplerMatrixVisitorHost.
 */
template<class B, class R1, class PX, class OX>
class StaticSamplerMatrixVisitorHost<B,empty_typelist,R1,PX,OX> {
public:
  static void accept(R1& rng, State<B,ON_HOST>& s, const int p, const PX& pax,
      OX& x) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class R1, class PX, class OX>
void bi::StaticSamplerMatrixVisitorHost<B,S,R1,PX,OX>::accept(R1& rng,
    State<B,ON_HOST>& s, const int p, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  front::samples(rng, s, p, pax, x);
  StaticSamplerMatrixVisitorHost<B,pop_front,R1,PX,OX>::accept(rng, s, p, pax,
      x);
}

#endif
