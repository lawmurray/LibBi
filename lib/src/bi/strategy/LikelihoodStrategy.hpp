/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_LIKELIHOODSTRATEGY_HPP
#define BI_STRATEGY_LIKELIHOODSTRATEGY_HPP

namespace bi {
/**
 * @internal
 *
 * Strategy for likelihood evaluation.
 *
 * @ingroup method_strategy
 *
 * @tparam X Node type.
 */
template<class X>
class LikelihoodStrategy {
public:
  /**
   * Likelihood calculation for o-node given parents.
   *
   * @tparam V1 parents type.
   * @tparam V2 Value type.
   * @tparam V3 %Result type.
   *
   * @param cox Coordinates of node.
   * @param pax %State of parents at the current time.
   * @param y Observation at the current time.
   * @param[out] l %Result. Likelihood of the observation.
   */
  template <class V1, class V2, class V3>
  static CUDA_FUNC_BOTH void l(const Coord& cox, const V1& pax, const V2& y,
      V3& l);

};
}

#include "../traits/likelihood_traits.hpp"
#include "GaussianLikelihoodStrategy.hpp"
#include "LogNormalLikelihoodStrategy.hpp"
#include "NoLikelihoodStrategy.hpp"

#include "boost/mpl/if.hpp"

template<class X>
template<class V1, class V2, class V3>
inline void bi::LikelihoodStrategy<X>::l(const Coord& cox, const V1& pax,
    const V2& y, V3& l) {
  using namespace boost::mpl;

  /* select strategy */
  typedef
    typename
    if_<is_gaussian_likelihood<X>,
        GaussianLikelihoodStrategy<X,V1,V2,V3>,
    typename
    if_<is_log_normal_likelihood<X>,
        LogNormalLikelihoodStrategy<X,V1,V2,V3>,
    /*else*/
        NoLikelihoodStrategy<X,V1,V2,V3>
    /*end*/
    >::type>::type strategy;

  /* execute strategy */
  strategy::l(cox, pax, y, l);
}

#endif
