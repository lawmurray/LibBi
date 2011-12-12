/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_DUPDATEVISITOR_CUH
#define BI_CUDA_UPDATER_DUPDATEVISITOR_CUH

#include "../../typelist/typelist.hpp"

namespace bi {
/**
 * @internal
 *
 * Device implementation of DUpdateVisitor.
 */
template<class B, class S, class V1, class V2>
class DUpdateVisitor<ON_DEVICE,B,S,V1,V2> {
public:
  /**
   * Update d-net.
   *
   * @param t Current time.
   * @param pax %State of parents.
   * @param tnxt Time to which to advance.
   * @param xnxt[out] Result.
   */
  static CUDA_FUNC_DEVICE void accept(const real t, const V1& pax,
      const real tnxt, V2& xnxt);
};

/**
 * @internal
 *
 * Device base case of DUpdateVisitor.
 */
template<class B, class V1, class V2>
class DUpdateVisitor<ON_DEVICE,B,empty_typelist,V1,V2> {
public:
  static CUDA_FUNC_DEVICE void accept(const real t, const V1& pax,
      const real tnxt, V2& xnxt) {
    //
  }
};

}

#include "../../typelist/front.hpp"
#include "../../typelist/pop_front.hpp"
#include "../../strategy/ForwardStrategy.hpp"

template<class B, class S, class V1, class V2>
inline void bi::DUpdateVisitor<bi::ON_DEVICE,B,S,V1,V2>::accept(
    const real t, const V1& pax, const real tnxt, V2& xnxt) {
  typedef typename front<S>::type front;
  typedef typename pop_front<S>::type pop_front;

  const int id = threadIdx.y;
  const int end = node_end<B,front>::value;
  if (id < end) {
    Coord cox = Coord::make<B,front>(id);
    ForwardStrategy<front>::f(cox, t, pax, tnxt, xnxt);
  } else {
    DUpdateVisitor<bi::ON_DEVICE,B,pop_front,V1,V2>::accept(t, pax, tnxt, xnxt);
  }
}

#endif
