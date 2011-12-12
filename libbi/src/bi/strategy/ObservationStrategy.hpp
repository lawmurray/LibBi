/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_OBSERVATIONSTRATEGY_HPP
#define BI_STRATEGY_OBSERVATIONSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for update of o-node.
 *
 * @ingroup method_strategy
 *
 * @tparam X Node type.
 */
template<class X>
class ObservationStrategy {
public:
  /**
   * Observation update. Update %state of o-node given states of parents.
   *
   * @tparam V1 Parents type.
   * @tparam V2 %Result type.
   *
   * @param r Gaussian random variate.
   * @param cox Coordinates of node.
   * @param pax %State of all parents.
   * @param[out] x %Result.
   */
  template <class V1, class V2>
  static CUDA_FUNC_BOTH void o(const real r, const Coord& cox,
      const V1& pax, V2& x);

};
}

#include "../traits/likelihood_traits.hpp"
#include "GaussianObservationStrategy.hpp"
#include "LogNormalObservationStrategy.hpp"
#include "NoObservationStrategy.hpp"

#include "boost/mpl/if.hpp"

template<class X>
template<class V1, class V2>
inline void bi::ObservationStrategy<X>::o(const real r, const Coord& cox,
    const V1& pax, V2& x) {
  using namespace boost::mpl;

  /* select strategy */
  typedef
    typename
    if_<is_gaussian_likelihood<X>,
        GaussianObservationStrategy<X,V1,V2>,
    typename
    if_<is_log_normal_likelihood<X>,
        LogNormalObservationStrategy<X,V1,V2>,
    /*else*/
        NoObservationStrategy<X,V1,V2>
    /*end*/
    >::type>::type strategy;

  /* execute strategy */
  strategy::o(r, cox, pax, x);
}

#endif
