/**
 * @file
 *
 * Generic (host and device) functions for distributions which are sampled
 * via the simulation of other more basic distributions.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RANDOM_GENERIC_HPP
#define BI_RANDOM_GENERIC_HPP

#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Generate a random number from a beta distribution with given parameters.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param alpha Shape.
 * @param beta Shape.
 *
 * @return The random number.
 */
template<class R, class T1>
CUDA_FUNC_BOTH T1 beta(R& rng, const T1 alpha = 1.0, const T1 beta = 1.0);

/**
 * Generate a random number from a inverse gamma distribution with given
 * parameters.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param alpha Shape.
 * @param beta Scale.
 *
 * @return The random number.
 */
template<class R, class T1>
CUDA_FUNC_BOTH T1 inverse_gamma(R& rng, const T1 alpha = 1.0, const T1 beta =
    1.0);

/**
 * Generate a random number from a negative binomial distribution with given
 * parameters.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param mu Mean.
 * @param k Shape.
 *
 * @return The random number.
 */
template<class R, class T1>
CUDA_FUNC_BOTH T1 negbin(R& rng, const T1 mu = 1.0, const T1 k = 1.0);

/**
 * Generate a random number from a binomial distribution with given
 * parameters.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param n Size.
 * @param p Probability.
 *
 * @return The random number.
 */
template<class R, class T1, class T2>
CUDA_FUNC_BOTH T1 binomial(R& rng, T1 n = 1.0, T2 p = 0.5);

}

template<class R, class T1>
inline T1 bi::beta(R& rng, const T1 alpha, const T1 beta) {
  /* pre-condition */
  BI_ASSERT(alpha > static_cast<T1>(0.0) && beta > static_cast<T1>(0.0));

  const T1 x = rng.gamma(alpha, static_cast<T1>(1.0));
  const T1 y = rng.gamma(beta, static_cast<T1>(1.0));

  return x / (x + y);
}

template<class R, class T1>
inline T1 bi::inverse_gamma(R& rng, const T1 alpha, const T1 beta) {
  /* pre-condition */
  BI_ASSERT(alpha > static_cast<T1>(0.0) && beta > static_cast<T1>(0.0));

  const T1 x = rng.gamma(alpha, static_cast<T1>(1.0)/beta);

  return static_cast<T1>(1.0)/x;
}

template<class R, class T1>
inline T1 bi::negbin(R& rng, const T1 mu, const T1 k) {
  /* pre-condition */
  BI_ASSERT(mu >= static_cast<T1>(0.0) && k >= static_cast<T1>(0.0));

  T1 u;

  if (mu > 0) {
    const T1 x = rng.gamma(k, mu/k);

    u = static_cast<T1>(rng.poisson(x));
  } else {
    u = 0;
  }

  return u;
}

template<class R, class T1, class T2>
inline T1 bi::binomial(R& rng, T1 n, T2 p) {
  /* pre-condition */
  BI_ASSERT(n >= static_cast<T1>(0.0) &&
            p >= static_cast<T2>(0.0) && p <= static_cast<T2>(1.0));

  T1 u;

  u = static_cast<T1>(rng.binomial(n, p));

  return u;
}
#endif
