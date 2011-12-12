/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_STATICSTRATEGY_HPP
#define BI_STRATEGY_STATICSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for update of s-node.
 *
 * @ingroup method_strategy
 *
 * @tparam X Node type.
 */
template<class X>
class StaticStrategy {
public:
  /**
   * Static update. Update %state of s-node given states of parents.
   *
   * @tparam V1 Parents type.
   * @tparam V2 %Result type.
   *
   * @param cox Coordinates of node.
   * @param pax %State of all parents.
   * @param[out] x %Result.
   */
  template<class V1, class V2>
  static CUDA_FUNC_BOTH void s(const Coord& cox, const V1& pax, V2& x);

};
}

#include "../traits/static_traits.hpp"
#include "GenericStaticStrategy.hpp"
#include "NoStaticStrategy.hpp"

#include "boost/mpl/if.hpp"

template<class X>
template<class V1, class V2>
inline void bi::StaticStrategy<X>::s(const Coord& cox, const V1& pax,
    V2& x) {
  using namespace boost::mpl;

  /* select strategy */
  typedef
    typename
    if_<is_generic_static<X>,
        GenericStaticStrategy<X,V1,V2>,
    /*else*/
        NoStaticStrategy<X,V1,V2>
    /*end*/
    >::type strategy;

  /* execute strategy */
  strategy::s(cox, pax, x);
}

#endif
