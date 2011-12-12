/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_SUPDATEVISITOR_HPP
#define BI_UPDATER_SUPDATEVISITOR_HPP

namespace bi {
/**
 * @internal
 *
 * Visitor for updating s-net.
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam V1 Parents type.
 */
template<Location L, class B, class S, class V1>
class SUpdateVisitor {
public:
  /**
   * Update s-net.
   *
   * @param pax %State of parents.
   * @param xnxt[out] Results.
   */
  static void accept(const V1& pax);
};

/**
 * @internal
 *
 * Host implementation of SUpdateVisitor.
 */
template<class B, class S, class V1>
class SUpdateVisitor<ON_HOST,B,S,V1> {
public:
  static void accept(const V1& pax);
};

/**
 * @internal
 *
 * Device implementation of SUpdateVisitor.
 */
template<class B, class S, class V1>
class SUpdateVisitor<ON_DEVICE,B,S,V1> {
public:
  static CUDA_FUNC_DEVICE void accept(const V1& pax);
};

/**
 * @internal
 *
 * Host base case of SUpdateVisitor.
 */
template<class B, class V1>
class SUpdateVisitor<ON_HOST,B,empty_typelist,V1> {
public:
  static void accept(const V1& pax) {
    //
  }
};

/**
 * @internal
 *
 * Device base case of SUpdateVisitor.
 */
template<class B, class V1>
class SUpdateVisitor<ON_DEVICE,B,empty_typelist,V1> {
public:
  static CUDA_FUNC_DEVICE void accept(const V1& pax) {
    //
  }
};

}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../host/host.hpp"
#include "../strategy/StaticStrategy.hpp"

template<class B, class S, class V1>
inline void bi::SUpdateVisitor<bi::ON_HOST,B,S,V1>::accept(const V1& pax) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef SUpdateVisitor<ON_HOST,B,pop_front,V1> Visitor;

  real val;
  Coord cox;
  int id = node_start<B,front>::value;
  for (cox.z = 0; cox.z < (int)node_z_size<B,front>::value; ++cox.z) {
    for (cox.y = 0; cox.y < (int)node_y_size<B,front>::value; ++cox.y) {
      for (cox.x = 0; cox.x < (int)node_x_size<B,front>::value; ++cox.x, ++id) {
        StaticStrategy<front>::s(cox, pax, val);
        hostSState(pax.p, id) = val;
      }
    }
  }
  Visitor::accept(pax);
}

#endif
