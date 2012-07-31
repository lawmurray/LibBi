/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RANDOM_RNG_HPP
#define BI_HOST_RANDOM_RNG_HPP

#include "boost/random/mersenne_twister.hpp"

namespace bi {
/**
 * Pseudorandom number generator, on host.
 *
 * @ingroup math_rng
 *
 * Uses the Mersenne Twister algorithm for generating pseudorandom variates,
 * as implemented in Boost.Random.
 *
 * @section RngHost_references References
 *
 * @anchor Matsumoto1998 Matsumoto, M. and Nishimura,
 * T. Mersenne Twister: A 623-dimensionally equidistributed
 * uniform pseudorandom number generator. <i>ACM Transactions on
 * Modeling and Computer Simulation</i>, <b>1998</b>, 8, 3-30.
 */
class RngHost {
public:
  /**
   * Seed random number generator.
   *
   * @param seed Seed value.
   */
  void seed(const unsigned seed);

  /**
   * Generate a random integer from a uniform distribution over a
   * given interval.
   *
   * @tparam T1 Scalar type.
   *
   * @param lower Lower bound on the interval.
   * @param upper Upper bound on the interval.
   *
   * @return The random integer, >= @p lower and <= @p upper.
   */
  template<class T1>
  T1 uniformInt(const T1 lower = 0, const T1 upper = 1);

  /**
   * Generate a random number from a multinomial distribution with given
   * probabilities.
   *
   * @tparam V1 Vector type.
   *
   * @param ps Log-probabilities. Need not be normalised to 1.
   *
   * @return Random index between @c 0 and <tt>ps.size() - 1</tt>, selected
   * according to the non-normalised probabilities given in @c ps.
   */
  template<class V1>
  typename V1::difference_type multinomial(const V1 ps);

  /**
   * Generate a random number from a uniform distribution over a
   * given interval.
   *
   * @tparam T1 Scalar type.
   *
   * @param lower Lower bound on the interval.
   * @param upper Upper bound on the interval.
   *
   * @return The random number.
   */
  template<class T1>
  T1 uniform(const T1 lower = 0.0, const T1 upper = 1.0);

  /**
   * Generate a random number from a Gaussian distribution with a
   * given mean and standard deviation.
   *
   * @tparam T1 Scalar type.
   *
   * @param mu Mean of the distribution.
   * @param sigma Standard deviation of the distribution.
   *
   * @return The random number. If the standard deviation is zero, returns
   * the mean.
   */
  template<class T1>
  T1 gaussian(const T1 mu = 0.0, const T1 sigma = 1.0);

  /**
   * Generate a random number from a gamma distribution with a given shape
   * and scale.
   *
   * @tparam T1 Scalar type.
   *
   * @param alpha Shape.
   * @param beta Scale.
   *
   * @return The random number.
   */
  template<class T1>
  T1 gamma(const T1 alpha = 1.0, const T1 beta = 1.0);

private:
  /**
   * Random number generator type.
   */
  typedef boost::mt19937 rng_type;

  /**
   * Random number generator.
   */
  rng_type rng;
};
}

#include "../../misc/location.hpp"
#include "../../misc/omp.hpp"
#include "../../math/sim_temp_vector.hpp"

#include "boost/random/uniform_int.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/gamma_distribution.hpp"
#include "boost/random/variate_generator.hpp"

#include "thrust/binary_search.h"

inline void bi::RngHost::seed(const unsigned seed) {
  rng.seed(seed);
}

template<class T1>
inline T1 bi::RngHost::uniformInt(const T1 lower, const T1 upper) {
  /* pre-condition */
  assert (upper >= lower);

  typedef boost::uniform_int<T1> dist_type;

  dist_type dist(lower, upper);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class V1>
inline typename V1::difference_type bi::RngHost::multinomial(const V1 ps) {
  /* pre-condition */
  assert (ps.size() > 0);

  typedef boost::uniform_real<typename V1::value_type> dist_type;

  typename sim_temp_vector<V1>::type Ps(ps.size());
  inclusive_scan_sum_expu(ps, Ps);

  dist_type dist(0.0, *(Ps.end() - 1));
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return thrust::lower_bound(Ps.begin(), Ps.end(), gen()) - Ps.begin();
}

template<class T1>
inline T1 bi::RngHost::uniform(const T1 lower, const T1 upper) {
  /* pre-condition */
  assert (upper >= lower);

  typedef boost::uniform_real<T1> dist_type;

  dist_type dist(lower, upper);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class T1>
inline T1 bi::RngHost::gaussian(const T1 mu, const T1 sigma) {
  /* pre-condition */
  assert (sigma >= 0.0);

  typedef boost::normal_distribution<T1> dist_type;

  dist_type dist(mu, sigma);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class T1>
inline T1 bi::RngHost::gamma(const T1 alpha, const T1 beta) {
  /* pre-condition */
  assert (alpha > 0.0 && beta > 0.0);

  typedef boost::gamma_distribution<T1> dist_type;

  dist_type dist(alpha);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return beta*gen();
}

#endif
