/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_OUPDATEVISITOR_CUH
#define BI_CUDA_UPDATER_OUPDATEVISITOR_CUH

#include "../../typelist/typelist.hpp"

namespace bi {
/**
 * @internal
 *
 * Device implementation of OUpdateVisitor.
 */
template<class B, class S, class V1, class V2, class V3>
class OUpdateVisitor<ON_DEVICE,B,S,V1,V2,V3> {
public:
  /**
   * Update o-net.
   *
   * @param id Id of the o-node to update.
   * @param r Gaussian random variate.
   * @param pax Parents.
   * @param[out] x Result.
   */
  static CUDA_FUNC_DEVICE void accept(const int id, const real r,
      const V2& pax, V3& x);
};

/**
 * @internal
 *
 * Device base case of OUpdateVisitor.
 */
template<class B, class V1, class V2, class V3>
class OUpdateVisitor<ON_DEVICE,B,empty_typelist,V1,V2,V3> {
public:
  static CUDA_FUNC_DEVICE void accept(const int id, const real r,
      const V2& pax, V3& x) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../strategy/ObservationStrategy.hpp"
#include "../constant.cuh"
#include "../global.cuh"
#include "../../state/Pa.hpp"

template<class B, class S, class V1, class V2, class V3>
inline void bi::OUpdateVisitor<bi::ON_DEVICE,B,S,V1,V2,V3>::accept(const int id,
    const real r, const V2& pax, V3& x) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  const int end = node_end<B,front>::value;
  if (id < end) {
    Coord cox = Coord::make<B,front>(id);
    ObservationStrategy<front>::o(r, cox, pax, x);
  } else {
    OUpdateVisitor<bi::ON_DEVICE,B,pop_front,V1,V2,V3>::accept(id, r, pax, x);
  }
}

#endif
