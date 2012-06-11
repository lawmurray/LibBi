/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2589 $
 * $Date: 2012-05-23 13:15:11 +0800 (Wed, 23 May 2012) $
 */
#ifndef BI_HOST_RANDOM_RNG_HPP
#define BI_HOST_RANDOM_RNG_HPP

#include "../../misc/location.hpp"
#include "../../misc/omp.hpp"
#include "../../math/sim_temp_vector.hpp"

#include "boost/random/uniform_int.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/gamma_distribution.hpp"
#include "boost/random/variate_generator.hpp"

#include "thrust/binary_search.h"

inline void bi::Rng<bi::ON_HOST>::seed(const unsigned seed) {
  rng.seed(seed);
}

template<class T1>
inline T1 bi::Rng<bi::ON_HOST>::uniformInt(const T1 lower, const T1 upper) {
  /* pre-condition */
  assert (upper >= lower);

  typedef boost::uniform_int<T1> dist_type;

  dist_type dist(lower, upper);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class V1>
inline typename V1::difference_type bi::Rng<bi::ON_HOST>::multinomial(const V1 ps) {
  /* pre-condition */
  assert (ps.size() > 0);

  typedef boost::uniform_real<typename V1::value_type> dist_type;

  typename sim_temp_vector<V1>::type Ps(ps.size());
  inclusive_scan_sum_exp(ps, Ps);

  dist_type dist(0.0, *(Ps.end() - 1));
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return thrust::lower_bound(Ps.begin(), Ps.end(), gen()) - Ps.begin();
}

template<class T1>
inline T1 bi::Rng<bi::ON_HOST>::uniform(const T1 lower, const T1 upper) {
  /* pre-condition */
  assert (upper >= lower);

  typedef boost::uniform_real<T1> dist_type;

  dist_type dist(lower, upper);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class T1>
inline T1 bi::Rng<bi::ON_HOST>::gaussian(const T1 mu, const T1 sigma) {
  /* pre-condition */
  assert (sigma >= 0.0);

  typedef boost::normal_distribution<T1> dist_type;

  dist_type dist(mu, sigma);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class T1>
inline T1 bi::Rng<bi::ON_HOST>::gamma(const T1 alpha, const T1 beta) {
  /* pre-condition */
  assert (alpha > 0.0 && beta > 0.0);

  typedef boost::gamma_distribution<T1> dist_type;

  dist_type dist(alpha);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return beta*gen();
}

#endif
