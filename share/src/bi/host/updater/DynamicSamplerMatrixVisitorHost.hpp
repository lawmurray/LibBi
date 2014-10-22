/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_DYNAMICSAMPLERMATRIXVISITORHOST_HPP
#define BI_HOST_UPDATER_DYNAMICSAMPLERMATRIXVISITORHOST_HPP

namespace bi {
/**
 * Visitor for DynamicSamplerHost.
 */
template<class B, class S, class R1, class PX, class OX>
class DynamicSamplerMatrixVisitorHost {
public:
  template<class T1>
  static void accept(R1& rng, const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      const int p, const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of DynamicSamplerMatrixVisitorHost.
 */
template<class B, class R1, class PX, class OX>
class DynamicSamplerMatrixVisitorHost<B,empty_typelist,R1,PX,OX> {
public:
  template<class T1>
  static void accept(R1& rng, const T1 t1, const T1 t2, State<B,ON_HOST>& s,
      const int p, const PX& pax, OX& x) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class R1, class PX, class OX>
template<class T1>
void bi::DynamicSamplerMatrixVisitorHost<B,S,R1,PX,OX>::accept(R1& rng,
    const T1 t1, const T1 t2, State<B,ON_HOST>& s, const int p,
    const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  front::samples(rng, t1, t2, s, p, pax, x);
  DynamicSamplerMatrixVisitorHost<B,pop_front,R1,PX,OX>::accept(rng, t1, t2,
      s, p, pax, x);
}

#endif
