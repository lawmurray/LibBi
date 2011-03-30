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

namespace bi {
/**
 * @internal
 *
 * Visitor for calculating LogLikelihood on device.
 *
 * @tparam L Location.
 * @tparam B Model type.
 * @tparam S Type list.
 * @tparam V1 Ids list type.
 * @tparam V2 Parents type.
 * @tparam V3 Result type.
 */
template<Location L, class B, class S, class V1, class V2, class V3>
class LUpdateVisitor {
public:
  /**
   * Calculate log-likelihood.
   *
   * @param ids Ids observed o-nodes.
   * @param i Current index into @p ids.
   * @param pax State of parents.
   * @param[in,out] Log-likelihood.
   */
  static void accept(const V1& ids, const typename V1::size_type i,
      const V2& pax, V3& l);
};

/**
 * @internal
 *
 * Host implementation of LUpdateVisitor.
 */
template<class B, class S, class V1, class V2, class V3>
class LUpdateVisitor<ON_HOST,B,S,V1,V2,V3> {
public:
  static void accept(const V1& ids, const typename V1::size_type i,
      const V2& pax, V3& l);
};

/**
 * @internal
 *
 * Host base case of LUpdateVisitor.
 */
template<class B, class V1, class V2, class V3>
class LUpdateVisitor<ON_HOST,B,empty_typelist,V1,V2,V3> {
public:
  static void accept(const V1& ids, const typename V1::size_type i,
      const V2& pax, V3& l) {
    //
  }
};

/**
 * @internal
 *
 * Device implementation of LUpdateVisitor.
 */
template<class B, class S, class V1, class V2, class V3>
class LUpdateVisitor<ON_DEVICE,B,S,V1,V2,V3> {
public:
  /**
   * Calculate log-likelihood.
   *
   * @param id Id of the o-node corresponding to the observation.
   * @param pax State of parents.
   * @param y Observation.
   * @param[out] Likelihood.
   */
  static CUDA_FUNC_DEVICE void accept(const int id, const V1& pax,
      const V2& y, V3& l);
};

/**
 * @internal
 *
 * Device base case of LUpdateVisitor.
 */
template<class B, class V1, class V2, class V3>
class LUpdateVisitor<ON_DEVICE,B,empty_typelist,V1,V2,V3> {
public:
  static CUDA_FUNC_DEVICE void accept(const int id, const V1& pax,
      const V2& y, V3& l) {
    //
  }
};
}

#include "../typelist/front.hpp"
#include "../typelist/pop_front.hpp"
#include "../strategy/LogLikelihoodStrategy.hpp"

template<class B, class S, class V1, class V2, class V3>
void bi::LUpdateVisitor<bi::ON_HOST,B,S,V1,V2,V3>::accept(const V1& ids,
    const typename V1::size_type i, const V2& pax, V3& l) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  if (i < ids.size()) {
    V3 y, val;
    if (ids[i] == node_id<B,front>::value) {
      /* observation of this variable available */
      Coord cox;
      for (cox.z = 0; cox.z < node_z_size<B,front>::value; ++cox.z) {
        for (cox.y = 0; cox.y < node_y_size<B,front>::value; ++cox.y) {
          for (cox.x = 0; cox.x < node_x_size<B,front>::value; ++cox.x) {
            y = host_fetch<B,front,0,0,0>(pax.p, cox);
            LogLikelihoodStrategy<front>::ll(cox, pax, y, val);
            l += val;
          }
        }
      }
      LUpdateVisitor<ON_HOST,B,pop_front,V1,V2,V3>::accept(ids, i + 1, pax, l);
    } else {
      LUpdateVisitor<ON_HOST,B,pop_front,V1,V2,V3>::accept(ids, i, pax, l);
    }
  }
}

#ifdef __CUDACC__
#include "../cuda/updater/LUpdateVisitor.cuh"
#endif

#endif
