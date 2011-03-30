/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_DUPDATEVISITOR_HPP
#define BI_UPDATER_DUPDATEVISITOR_HPP

namespace bi {
/**
 * @internal
 *
 * Visitor for updating d-net.
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam V1 Parents type.
 * @tparam V2 %Result type.
 */
template<Location L, class B, class S, class V1, class V2>
class DUpdateVisitor {
public:
  /**
   * Update d-net.
   *
   * @param t Current time.
   * @param pax %State of parents.
   * @param tnxt Time to which to advance.
   * @param xnxt[out] Results.
   */
  static void accept(const real t, const V1& pax,
      const real tnxt, V2* xnxt);
};

/**
 * @internal
 *
 * Host implementation of DUpdateVisitor.
 */
template<class B, class S, class V1, class V2>
class DUpdateVisitor<ON_HOST,B,S,V1,V2> {
public:
  static void accept(const real t, const V1& pax,
      const real tnxt, V2* xnxt);
};

/**
 * @internal
 *
 * Host base case of DUpdateVisitor.
 */
template<class B, class V1, class V2>
class DUpdateVisitor<ON_HOST,B,empty_typelist,V1,V2> {
public:
  static void accept(const real t, const V1& pax,
      const real tnxt, V2* xnxt) {
    //
  }
};

}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../host/host.hpp"
#include "../strategy/ForwardStrategy.hpp"

template<class B, class S, class V1, class V2>
inline void bi::DUpdateVisitor<bi::ON_HOST,B,S,V1,V2>::accept(
    const real t, const V1& pax, const real tnxt, V2* xnxt) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  Coord cox;
  int id = node_start<B,front>::value;
  for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
    for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
      for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
        ForwardStrategy<front>::f(cox, t, pax, tnxt, xnxt[id]);
      }
    }
  }
  DUpdateVisitor<ON_HOST,B,pop_front,V1,V2>::accept(t, pax, tnxt, xnxt);
}

#ifdef __CUDACC__
#include "../cuda/updater/DUpdateVisitor.cuh"
#endif

#endif
