/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_UPDATER_LUPDATEVISITOR_HPP
#define BI_UPDATER_LUPDATEVISITOR_HPP

#include "../typelist/typelist.hpp"
#include "../buffer/SparseMask.hpp"

namespace bi {
/**
 * @internal
 *
 * Visitor for calculating log-likelihood.
 *
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam V1 Parents type.
 * @tparam T1 Observation type.
 * @tparam T2 Result type.
 */
template<class B, class S, class V1, class T1, class T2>
class LUpdateVisitor {
public:
  /**
   * Calculate log-likelihood.
   *
   * @param id Id of the o-node corresponding to the observation.
   * @param coord Spatial coordinate of the observation.
   * @param pax State of parents.
   * @param[out] Likelihood.
   */
  static CUDA_FUNC_BOTH void accept(const int id, const Coord& cox,
      const V1& pax, T1 y, T2& l);
};

/**
 * @internal
 *
 * Base case of LUpdateVisitor.
 */
template<class B, class V1, class T1, class T2>
class LUpdateVisitor<B,empty_typelist,V1,T1,T2> {
public:
  static CUDA_FUNC_BOTH void accept(const int id, const Coord& cox,
      const V1& pax, T1 y, T2& l) {
    //
  }
};
}

#include "../host/host.hpp"
#include "../host/const_host.hpp"
#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../strategy/LogLikelihoodStrategy.hpp"

template<class B, class S, class V1, class T1, class T2>
void bi::LUpdateVisitor<B,S,V1,T1,T2>::accept(const int id,
    const Coord& cox, const V1& pax, T1 y, T2& l) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef LUpdateVisitor<B,pop_front,V1,T1,T2> Visitor;

  if (id == node_id<B,front>::value) {
    LogLikelihoodStrategy<front>::ll(cox, pax, y, l);
  } else {
    Visitor::accept(id, cox, pax, y, l);
  }
}

#endif
