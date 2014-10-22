/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_UPDATER_STATICUPDATERMATRIXVISITORHOST_HPP
#define BI_HOST_UPDATER_STATICUPDATERMATRIXVISITORHOST_HPP

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
class StaticUpdaterMatrixVisitorHost {
public:
  /**
   * Update.
   *
   * @param s State.
   * @param p Trajectory id.
   * @param pax Parents.
   * @param[out] x Output.
   */
  static void accept(State<B,ON_HOST>& s, const int p, const PX& pax, OX& x);
};

/**
 * @internal
 *
 * Base case of StaticUpdaterMatrixVisitorHost.
 */
template<class B, class PX, class OX>
class StaticUpdaterMatrixVisitorHost<B,empty_typelist,PX,OX> {
public:
  static void accept(State<B,ON_HOST>& s, const int p, const PX& pax, OX& x) {
    //
  }
};
}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"

template<class B, class S, class PX, class OX>
inline void bi::StaticUpdaterMatrixVisitorHost<B,S,PX,OX>::accept(
    State<B,ON_HOST>& s, const int p, const PX& pax, OX& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  front::simulates(s, p, pax, x);
  StaticUpdaterMatrixVisitorHost<B,pop_front,PX,OX>::accept(s, p, pax, x);
}

#endif
