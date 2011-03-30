/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_SUPDATEVISITOR_CUH
#define BI_CUDA_UPDATER_SUPDATEVISITOR_CUH

#include "../../typelist/typelist.hpp"
#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../strategy/StaticStrategy.hpp"

template<class B, class S, class V1>
inline void bi::SUpdateVisitor<bi::ON_DEVICE,B,S,V1>::accept(const V1& pax) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  real val;
  Coord cox;
  int id = node_start<B,front>::value;
  for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
    for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
      for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x, ++id) {
        StaticStrategy<front>::s(cox, pax, val);
        globalSState(pax.p, id) = val;
      }
    }
  }
  SUpdateVisitor<bi::ON_DEVICE,B,pop_front,V1>::accept(pax);
}

#endif
