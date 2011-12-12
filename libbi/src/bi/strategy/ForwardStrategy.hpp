/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_FORWARDSTRATEGY_HPP
#define BI_STRATEGY_FORWARDSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for forward update of node through time.
 *
 * @ingroup method_strategy
 *
 * @tparam X Node type.
 */
template<class X>
class ForwardStrategy {
public:
  /**
   * Forward update. Update %state of d- or c-node given parents.
   *
   * @tparam T Time type.
   * @tparam V1 Parents type.
   * @tparam V2 %Result type.
   *
   * @param cox Coordinates of the node.
   * @param t The current time.
   * @param pax %State of parents at the current time.
   * @param tnxt The next time.
   * @param[out] xnxt %Result. Updated %state of node at time @p tnxt.
   */
  template <class T, class V1, class V2>
  static CUDA_FUNC_BOTH void f(const Coord& cox, const T t, const V1& pax,
      const T tnxt, V2& xnxt);

};
}

#include "../traits/forward_traits.hpp"
#include "GenericForwardStrategy.hpp"
#include "ODEForwardStrategy.hpp"
#include "NoForwardStrategy.hpp"

#include "boost/mpl/if.hpp"

template<class X>
template<class T, class V1, class V2>
inline void bi::ForwardStrategy<X>::f(const Coord& cox, const T t,
    const V1& pax, const T tnxt, V2& xnxt) {
  using namespace boost::mpl;

  /* select strategy */
  typedef
    typename
    if_<is_generic_forward<X>,
        GenericForwardStrategy<X,T,V1,V2>,
    typename
    if_<is_ode_forward<X>,
        ODEForwardStrategy<X,T,V1,V2>,
    /*else*/
        NoForwardStrategy<X,T,V1,V2>
    /*end*/
    >::type>::type strategy;

  /* execute strategy */
  strategy::f(cox, t, pax, tnxt, xnxt);
}

#endif
