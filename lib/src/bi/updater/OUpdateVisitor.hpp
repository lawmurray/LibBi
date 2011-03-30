/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_OUPDATEVISITOR_HPP
#define BI_UPDATER_OUPDATEVISITOR_HPP

#include "../typelist/typelist.hpp"

namespace bi {
/**
 * @internal
 *
 * Visitor for updating o-nodes on host.
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam V1 Integral vector type.
 * @tparam V2 Parents type.
 * @tparam V3 %Result type.
 */
template<Location L, class B, class S, class V1, class V2, class V3>
class OUpdateVisitor {
public:
  /**
   * Update o-net.
   *
   * @param ids Ids of the o-nodes to update.
   * @param pax Parents.
   * @param xnxt Results.
   * @param i Current index into @p ids.
   */
  static void accept(const V1& ids, const V2& pax, V3* xnxt,
      const int i = 0);
};

/**
 * @internal
 *
 * Host implementation of OUpdateVisitor.
 */
template<class B, class S, class V1, class V2, class V3>
class OUpdateVisitor<ON_HOST,B,S,V1,V2,V3> {
public:
  static void accept(const V1& ids, const V2& pax, V3* xnxt,
      const int i = 0);
};

/**
 * @internal
 *
 * Host base case of OUpdateVisitor.
 */
template<class B, class V1, class V2, class V3>
class OUpdateVisitor<ON_HOST,B,empty_typelist,V1,V2,V3> {
public:
  static void accept(const V1& ids, const V2& pax, V3* xnxt,
      const int i) {
    //
  }
};

}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../strategy/ObservationStrategy.hpp"
#include "../state/Pa.hpp"

template<class B, class S, class V1, class V2, class V3>
inline void bi::OUpdateVisitor<bi::ON_HOST,B,S,V1,V2,V3>::accept(const V1& ids,
    const V2& pax, V3* xnxt, const int i) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef OUpdateVisitor<ON_HOST,B,pop_front,V1,V2,V3> Visitor;

  if (i < ids.size()) {
    if (node_id<B,front>::value == ids[i]) {
      real x, r;
      Coord cox;
      int id = node_start<B,front>::value;
      for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
        for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
          for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
            r = hostORState(pax.p, i);
            ObservationStrategy<front>::o(r, cox, pax, x);
            xnxt[i] = x;
          }
        }
      }
      Visitor::accept(ids, pax, xnxt, i + 1);
    } else {
      Visitor::accept(ids, pax, xnxt, i);
    }
  }
}

#ifdef __CUDACC__
#include "../cuda/updater/OUpdateVisitor.cuh"
#endif

#endif
