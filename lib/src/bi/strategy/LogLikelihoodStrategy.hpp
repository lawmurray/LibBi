/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_LOGLIKELIHOODSTRATEGY_HPP
#define BI_STRATEGY_LOGLIKELIHOODSTRATEGY_HPP

#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for log-likelihood evaluation.
 *
 * @ingroup method_strategy
 *
 * @tparam X Node type.
 */
template<class X>
class LogLikelihoodStrategy {
public:
  /**
   * Log-likelihood calculation given parents.
   *
   * @tparam V1 parents type.
   * @tparam V2 Value type.
   * @tparam V3 Result type.
   *
   * @param cox Coordinates of node.
   * @param pax State of parents at the current time.
   * @param y Observation at the current time.
   * @param[out] l Result. Log-likelihood of the observation.
   */
  template <class V1, class V2, class V3>
  static CUDA_FUNC_BOTH void ll(const Coord& cox, const V1& pax, const V2& y,
      V3& l);

};
}

#include "../traits/likelihood_traits.hpp"
#include "GaussianLogLikelihoodStrategy.hpp"
#include "LogNormalLogLikelihoodStrategy.hpp"
#include "NoLogLikelihoodStrategy.hpp"

#include "boost/mpl/if.hpp"

template<class X>
template<class V1, class V2, class V3>
inline void bi::LogLikelihoodStrategy<X>::ll(const Coord& cox, const V1& pax,
    const V2& y, V3& ll) {
  using namespace boost::mpl;

  /* select strategy */
  typedef
    typename
    if_<is_gaussian_likelihood<X>,
        GaussianLogLikelihoodStrategy<X,V1,V2,V3>,
    typename
    if_<is_log_normal_likelihood<X>,
        LogNormalLogLikelihoodStrategy<X,V1,V2,V3>,
    /*else*/
        NoLogLikelihoodStrategy<X,V1,V2,V3>
    /*end*/
    >::type>::type strategy;

  /* execute strategy */
  strategy::ll(cox, pax, y, ll);
}

#endif
