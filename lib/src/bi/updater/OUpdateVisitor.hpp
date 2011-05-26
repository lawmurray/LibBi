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
#include "../buffer/SparseMask.hpp"

namespace bi {
/**
 * @internal
 *
 * Visitor for updating o-nodes on host.
 *
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam V1 Parents type.
 * @tparam T1 Scalar type.
 * @tparam T2 Scalar type.
 */
template<class B, class S, class V1, class T1, class T2>
class OUpdateVisitor {
public:
  /**
   * Update o-net.
   *
   * @param id Id of the o-node.
   * @param coord Spatial coordinate of the o-node.
   * @param pax State of parents.
   * @param r Associated or-node variate.
   * @param[out] o Predicted observation.
   */
  static CUDA_FUNC_BOTH void accept(const int id, const Coord& cox,
      const V1& pax, const T1 r, T2& o);
};

/**
 * @internal
 *
 * Base case of OUpdateVisitor.
 */
template<class B, class V1, class T1, class T2>
class OUpdateVisitor<B,empty_typelist,V1,T1,T2> {
public:
  static CUDA_FUNC_BOTH void accept(const int id, const Coord& cox,
      const V1& pax, const T1 r, T2& o) {
    //
  }
};

}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../strategy/ObservationStrategy.hpp"
#include "../state/Pa.hpp"

template<class B, class S, class V1, class T1, class T2>
inline void bi::OUpdateVisitor<B,S,V1,T1,T2>::accept(const int id,
    const Coord& cox, const V1& pax, const T1 r, T2& o) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef OUpdateVisitor<B,pop_front,V1,T1,T2> Visitor;

  if (id == node_id<B,front>::value) {
    ObservationStrategy<front>::o(r, cox, pax, o);
  } else {
    Visitor::accept(id, cox, pax, r, o);
  }
}

#endif
