/**
 * @file
 *
 * Truncated gaussian simluation. This largely copied from the
 * code for the 'truncnorm' R package, written by
 * Olaf Mersmann, Heike Trautmann, Detlef Steuer and Björn Bornkamp
 * https://github.com/cran/truncnorm/blob/master/src/rtruncnorm.c
 *
 * @author Sebastian Funk <sebastian.funk@lshtm.ac.uk>
 * $Rev$
 * $Date$
 */

#ifndef BI_TRUNCATED_NORMAL_HPP
#define BI_TRUNCATED_NORMAL_HPP

#define _t1 0.15
#define _t2 2.18
#define _t3 0.725
#define _t4 0.45

#include "generic.hpp"

namespace bi {

template<class R, class T1>
CUDA_FUNC_BOTH T1 ers_a_inf(R& rng, const T1 a);

template<class R, class T1>
CUDA_FUNC_BOTH T1 ers_a_b(R& rng, const T1 a, const T1 b);

template<class R, class T1>
CUDA_FUNC_BOTH T1 nrs_a_b(R& rng, const T1 a, const T1 b);

template<class R, class T1>
CUDA_FUNC_BOTH T1 nrs_a_inf(R& rng, const T1 a);

template<class R, class T1>
CUDA_FUNC_BOTH T1 hnrs_a_b(R& rng, const T1 a, const T1 b);

template<class T1>
CUDA_FUNC_BOTH T1 std_normal_dens(const T1 x);

template<class R, class T1>
CUDA_FUNC_BOTH T1 urs_a_b(R& rng, const T1 a, const T1 b);

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
}

/* Exponential rejection sampling (a,inf) */
template<class R, class T1>
T1 bi::ers_a_inf(R& rng, const T1 a) {
  T1 x, rho;
  do {
    x = bi::exponential(rng, a) + a;
    rho = bi::exp(-0.5 * bi::pow((x - a), 2));
  } while (rng.uniform(static_cast<T1>(0.0), static_cast<T1>(1.0)) > rho);
  return x;
}

/* Exponential rejection sampling (a,b) */
template<class R, class T1>
T1 bi::ers_a_b(R& rng, const T1 a, const T1 b) {
  T1 x, rho;
  do {
    x = bi::exponential(rng, a) + a;
    rho = bi::exp(-0.5 * bi::pow((x - a), 2));
  } while (rng.uniform(static_cast<T1>(0.0), static_cast<T1>(1.0)) > rho ||
           x > b);
  return x;
}

/* Normal rejection sampling (a,b) */
template<class R, class T1>
T1 bi::nrs_a_b(R& rng, const T1 a, const T1 b) {
  T1 x = -BI_INF;
  while (x < a || x > b) {
    x = rng.gaussian(static_cast<T1>(0.0), static_cast<T1>(1.0));
  }
  return x;
}

/* Normal rejection sampling (a,inf) */
template<class R, class T1>
T1 bi::nrs_a_inf(R& rng, const T1 a) {
  T1 x = -BI_INF;
  while (x < a) {
    x = rng.gaussian(static_cast<T1>(0.0), static_cast<T1>(1.0));
  }
  return x;
}

/* Half-normal rejection sampling */
template<class R, class T1>
T1 bi::hnrs_a_b(R& rng, const T1 a, const T1 b) {
  T1 x = a - static_cast<T1>(1.0);
  while (x < a || x > b) {
    x = rng.gaussian(static_cast<T1>(0.0), static_cast<T1>(1.0));
    x = bi::abs(x);
  }
  return x;
}

/* standard normal density */
template<class T1>
T1 bi::std_normal_dens(const T1 x) {
  return bi::exp(BI_REAL(-0.5)*
                 bi::pow(x, BI_REAL(2.0)) - BI_REAL(BI_HALF_LOG_TWO_PI));
}

/* Uniform rejection sampling */
template<class R, class T1>
T1 bi::urs_a_b(R& rng, const T1 a, const T1 b) {
  const T1 log_phi_a = bi::std_normal_dens(a);
  T1 x = static_cast<T1>(0.0), u = 0.0;

  /* Upper bound of normal density on [a, b] */
  const T1 ub = a < 0 && b > 0 ? BI_1_SQRT_TWO_PI : bi::exp(log_phi_a);
  do {
    x = rng.uniform(a, b);
  } while (rng.uniform(static_cast<T1>(0.0), static_cast<T1>(1.0)) * ub >
           bi::std_normal_dens(x));
  return x;
}

template<class R, class T1>
T1 bi::lower_truncated_gaussian(R& rng, const T1 lower, const T1 mu,
                                const T1 sigma) {
  BI_ASSERT(sigma > 0.0 || (sigma == 0 && lower <= mu));
  T1 u;

  if (sigma == 0) {
    u = mu;
  } else {
    const T1 alpha = (lower - mu) / sigma;
    if (alpha < _t4) {
      u = mu + sigma * bi::nrs_a_inf(rng, alpha);
    } else {
      u = mu + sigma * bi::ers_a_inf(rng, alpha);
    }
  }

  return u;
}

template<class R, class T1>
T1 bi::upper_truncated_gaussian(R& rng, const T1 upper, const T1 mu,
                                const T1 sigma) {
  BI_ASSERT(sigma > 0.0 || (sigma == 0 && mu <= upper));

  T1 u;

  if (sigma == 0) {
    u = mu;
  } else {
    T1 beta = (upper - mu) / sigma;
    /* Exploit symmetry: */
    u = mu - sigma *
      bi::lower_truncated_gaussian(rng, -beta,
                                   static_cast<T1>(0.0), static_cast<T1>(1.0));
  }
  return u;
}

template<class R, class T1>
T1 bi::truncated_gaussian(R& rng, const T1 lower, const T1 upper, const T1 mu,
                          const T1 sigma) {
  BI_ASSERT(upper >= lower);
  BI_ASSERT(sigma > 0.0 || (sigma == 0 && lower <= mu && mu <= upper));

  T1 u;
  if (sigma == 0) {
    u = mu;
  } else {
    const T1 alpha = (lower - mu) / sigma;
    const T1 beta = (upper - mu) / sigma;
    const T1 phi_a = bi::std_normal_dens(alpha);
    const T1 phi_b = bi::std_normal_dens(beta);

    if (alpha <= 0 && 0 <= beta) {
      if (phi_a <= _t1 || phi_b <= _t1) {
        u = mu + sigma * bi::nrs_a_b(rng, alpha, beta);
      } else {
        u = mu + sigma * bi::urs_a_b(rng, alpha, beta);
      }
    } else if (alpha > 0) {
      if (phi_a / phi_b <= _t2) {
        u = mu + sigma * bi::urs_a_b(rng, alpha, beta);
      } else {
        if (alpha < _t3) {
          u = mu + sigma * bi::hnrs_a_b(rng, alpha, beta);
        } else {
          u = mu + sigma * bi::ers_a_b(rng, alpha, beta);
        }
      }
    } else {
      if (phi_b / phi_a <= _t2) {
        u = mu - sigma * bi::urs_a_b(rng, -beta, -alpha);
      } else {
        if (beta > -_t3) {
          u = mu - sigma * bi::hnrs_a_b(rng, -beta, -alpha);
        } else {
          u = mu - sigma * bi::ers_a_b(rng, -beta, -alpha);
        }
      }
    }
  }

  return u;
}

#endif
