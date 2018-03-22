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
#include "../math/function.hpp"

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
 * Generate a random number from a Beta-binomial distribution with given
 * parameters.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param n Number of trials.
 * @param alpha Shape..
 * @param beta Scale.
 *
 * @return The random number.
 */
template<class R, class T1>
CUDA_FUNC_BOTH T1 betabin(R& rng, const T1 n = 1.0, const T1 alpha = 1.0,
                          const T1 beta = 1.0);

/**
 * Generate a random number from an exponential distribution with given
 * rate.
 *
 * @tparam R Random number generator type.
 * @tparam T1 Scalar type.
 *
 * @param[in,out] rng Random number generator.
 * @param lambda Rate.
 *
 * @return The random number.
 */
template<class R, class T1>
CUDA_FUNC_BOTH T1 exponential(R& rng, const T1 lambda = 1.0);

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

template<class R, class T1>
inline T1 bi::betabin(R& rng, const T1 n, const T1 alpha, const T1 beta) {
  /* pre-condition */
  BI_ASSERT(n >= static_cast<T1>(0.0) && alpha > static_cast<T1>(0.0) &&
            beta > static_cast<T1>(0.0));

  T1 u;

  if (n > 0) {
    const T1 p = bi::beta(rng, alpha, beta);

    u = static_cast<T1>(rng.binomial(n, p));
  } else {
    u = 0;
  }

  return u;
}

template<class R, class T1>
inline T1 bi::exponential(R& rng, const T1 lambda) {
  /* pre-condition */
  BI_ASSERT(lambda > static_cast<T1>(0.0));

  T1 u;

  const T1 x = rng.uniform(static_cast<T1>(0.0), static_cast<T1>(1.0));

  u = -log(1 - x) / lambda;

  return u;
}

#endif
