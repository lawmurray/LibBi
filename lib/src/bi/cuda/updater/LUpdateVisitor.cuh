/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_LUPDATEVISITOR_CUH
#define BI_CUDA_UPDATER_LUPDATEVISITOR_CUH

#include "../../typelist/typelist.hpp"
#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../strategy/LogLikelihoodStrategy.hpp"

template<class B, class S, class V1, class V2, class V3>
inline void bi::LUpdateVisitor<bi::ON_DEVICE,B,S,V1,V2,V3>::accept(
    const int id, const V1& pax, const V2& y, V3& l) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;
  typedef LUpdateVisitor<bi::ON_DEVICE,B,pop_front,V1,V2,V3> Visitor;

  const int end = node_end<B,front>::value;
  if (id < end) {
    Coord cox = Coord::make<B,front>(id);
    LogLikelihoodStrategy<front>::ll(cox, pax, y, l);
  } else {
    Visitor::accept(id, pax, y, l);
  }
}

#endif
