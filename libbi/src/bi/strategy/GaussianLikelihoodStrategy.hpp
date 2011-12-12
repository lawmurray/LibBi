/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_STRATEGY_GAUSSIANLIKELIHOODSTRATEGY_HPP
#define BI_STRATEGY_GAUSSIANLIKELIHOODSTRATEGY_HPP

#include "../traits/likelihood_traits.hpp"
#include "../math/pi.hpp"

namespace bi {
/**
 * @internal
 *
 * Strategy for Gaussian likelihood calculations, used by nodes with
 * #IS_GAUSSIAN_LIKELIHOOD (or its synonym, #IS_NORMAL_LIKELIHOOD) trait.
 *
 * @ingroup method_strategy
 *
 * @copydoc LikelihoodStrategy
 *
 * Note the following optimisations based on the traits of the node:
 *
 * @li If one or both of #HAS_ZERO_MU or #HAS_UNIT_SIGMA is true, fewer
 * operations need be performed in calculating the likelihood.
 */
template<class X, class V1, class V2, class V3>
class GaussianLikelihoodStrategy {
public:
  /**
   * @copydoc LikelihoodStrategy::l()
   */
  static CUDA_FUNC_BOTH void l(const Coord& cox, const V1& pax,
      const V2& y, V3& l) {
    real mu, sigma;

    if (has_zero_mu<X>::value) {
      if (has_unit_sigma<X>::value) {
        l = exp(REAL(-0.5)*CUDA_POW(y,2));
        ///@todo Make BI_INV_SQRT_TWO_PI and multiply
        l /= REAL(BI_SQRT_TWO_PI);
      } else {
        X::sigma(cox, pax, sigma);
        l = exp(REAL(-0.5)*CUDA_POW(sigma,-2)*CUDA_POW(y,2));
        l /= REAL(BI_SQRT_TWO_PI)*sigma; // normalise
      }
    } else {
      X::mu(cox, pax, mu);
      if (has_unit_sigma<X>::value) {
        l = exp(REAL(-0.5)*CUDA_POW(y - mu,2));
        l /= REAL(BI_SQRT_TWO_PI);
      } else {
        X::sigma(cox, pax, sigma);
        l = exp(REAL(-0.5)*CUDA_POW(sigma,-2)*pow(y - mu,2));
        l /= REAL(BI_SQRT_TWO_PI)*sigma; // normalise
      }
    }
  }

};

}

#endif
