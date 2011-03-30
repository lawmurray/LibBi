/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_LOGNORMALOBSERVATIONSTRATEGY_HPP
#define BI_STRATEGY_LOGNORMALOBSERVATIONSTRATEGY_HPP

#include "../cuda/cuda.hpp"
#include "../traits/likelihood_traits.hpp"
#include "../state/Coord.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for observation update with IS_LOG_NORMAL_LIKELIHOOD.
 *
 * @ingroup method_strategy
 *
 * @copydoc ObservationStrategy
 */
template<class X, class V1, class V2>
class LogNormalObservationStrategy {
public:
  /**
   * @copydoc ObservationStrategy::o()
   */
  static CUDA_FUNC_BOTH void o(const real r, const Coord& cox,
      const V1& pax, V2& x) {
    real mu, sigma;

    if (has_zero_mu<X>::value) {
      if (has_unit_sigma<X>::value) {
        x = CUDA_EXP(r);
      } else {
        X::sigma(cox, pax, sigma);
        x = CUDA_EXP(sigma*r);
      }
    } else {
      X::mu(cox, pax, mu);
      if (has_unit_sigma<X>::value) {
        x = CUDA_EXP(mu + r);
      } else {
        X::sigma(cox, pax, sigma);
        x = CUDA_EXP(mu + sigma*r);
      }
    }
  }

};
}

#endif
