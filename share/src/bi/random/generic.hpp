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
 * Generate a random number from a Gaussian distribution that is truncated
 * with a lower bound.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param lower Lower bound of the distribution.
 * @param mu Mean of the distribution.
 * @param sigma Standard deviation of the distribution.
 *
 * @return The random number.
 */
template<class R, class T1>
CUDA_FUNC_BOTH T1 lower_truncated_gaussian(R& rng, const T1 lower,
    const T1 mu = 0.0, const T1 sigma = 1.0);

/**
 * Generate a random number from a Gaussian distribution that is truncated
 * with a upper bound.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param upper Upper bound of the distribution.
 * @param mu Mean of the distribution.
 * @param sigma Standard deviation of the distribution.
 *
 * @return The random number.
 */
template<class R, class T1>
CUDA_FUNC_BOTH T1 upper_truncated_gaussian(R& rng, const T1 upper,
    const T1 mu = 0.0, const T1 sigma = 1.0);

/**
 * Generate a random number from a Gaussian distribution that is truncated
 * with both a lower and an upper bound.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param lower Lower bound of the distribution.
 * @param upper Upper bound of the distribution.
 * @param mu Mean of the distribution.
 * @param sigma Standard deviation of the distribution.
 *
 * @return The random number.
 */
template<class R, class T1>
CUDA_FUNC_BOTH T1 truncated_gaussian(R& rng, const T1 lower, const T1 upper,
    const T1 mu = 0.0, const T1 sigma = 1.0);

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
T1 bi::lower_truncated_gaussian(R& rng, const T1 lower, const T1 mu,
    const T1 sigma) {
  T1 u;
  do {
    u = rng.gaussian(mu, sigma);
  } while (u <= lower);

  return u;
}

template<class R, class T1>
T1 bi::upper_truncated_gaussian(R& rng, const T1 upper, const T1 mu,
    const T1 sigma) {
  T1 u;
  do {
    u = rng.gaussian(mu, sigma);
  } while (u >= upper);

  return u;
}

template<class R, class T1>
T1 bi::truncated_gaussian(R& rng, const T1 lower, const T1 upper, const T1 mu,
    const T1 sigma) {
  /* pre-conditions */
  BI_ASSERT(upper >= lower);

  T1 u;
  if (upper == lower) {
    u = upper;
  } else do {
    u = rng.gaussian(mu, sigma);
  } while (u <= lower || u >= upper);

  return u;
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
