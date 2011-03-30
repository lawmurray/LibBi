/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_GAUSSIANLOGLIKELIHOODSTRATEGY_HPP
#define BI_STRATEGY_GAUSSIANLOGLIKELIHOODSTRATEGY_HPP

#include "../traits/likelihood_traits.hpp"
#include "../math/pi.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for Gaussian log-likelihood calculations, used by nodes with
 * #IS_GAUSSIAN_LIKELIHOOD trait (or its synonym, #IS_NORMAL_LIKELIHOOD).
 *
 * @ingroup method_strategy
 *
 * @copydoc LogLikelihoodStrategy
 *
 * Note the following optimisations based on the traits of the node:
 *
 * @li If one or both of #HAS_ZERO_MU or #HAS_UNIT_SIGMA is true, fewer
 * operations need be performed in calculating the log-likelihood.
 */
template<class X, class V1, class V2, class V3>
class GaussianLogLikelihoodStrategy {
public:
  /**
   * @copydoc LogLikelihoodStrategy::ll()
   */
  static CUDA_FUNC_BOTH void ll(const Coord& cox, const V1& pax,
      const V2& y, V3& ll) {
    real mu, sigma;

    if (has_zero_mu<X>::value) {
      if (has_unit_sigma<X>::value) {
        ll = REAL(-0.5)*CUDA_POW(y,2);
        ll -= REAL(BI_HALF_LOG_TWO_PI);
      } else {
        X::sigma(cox, pax, sigma);
        ll = REAL(-0.5)*CUDA_POW(sigma,-2)*CUDA_POW(y,2);
        ll -= REAL(BI_HALF_LOG_TWO_PI) + CUDA_LOG(sigma);
      }
    } else {
      X::mu(cox, pax, mu);
      if (has_unit_sigma<X>::value) {
        ll = REAL(-0.5)*CUDA_POW(y - mu,2);
        ll -= REAL(BI_HALF_LOG_TWO_PI);
      } else {
        X::sigma(cox, pax, sigma);
        ll = REAL(-0.5)*CUDA_POW(y - mu,2)*CUDA_POW(sigma,-2);
        ll -= REAL(BI_HALF_LOG_TWO_PI) + CUDA_LOG(sigma);
      }
    }
  }

};

}

#endif
