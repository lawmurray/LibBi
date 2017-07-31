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
   * @copydoc Random::uniformInt
   */
  template<class T1>
  T1 uniformInt(const T1 lower = 0, const T1 upper = 1);

  /**
   * @copydoc Random::multinomial
   */
  template<class V1>
  typename V1::difference_type multinomial(const V1 lps);

  /**
   * @copydoc Random::uniform
   */
  template<class T1>
  T1 uniform(const T1 lower = 0.0, const T1 upper = 1.0);

  /**
   * @copydoc Random::gaussian
   */
  template<class T1>
  T1 gaussian(const T1 mu = 0.0, const T1 sigma = 1.0);

  /**
   * @copydoc Random::gamma
   */
  template<class T1>
  T1 gamma(const T1 alpha = 1.0, const T1 beta = 1.0);

  /**
   * @copydoc Random::poisson
   */
  template<class T1>
  T1 poisson(const T1 lambda = 1.0);

  /**
   * @copydoc Random::binomial
   */
  template<class T1, class T2>
  T1 binomial(const T1 n = 1.0, const T2 p = 0.5);

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

#include "../../misc/omp.hpp"
#include "../../math/sim_temp_vector.hpp"

#include "boost/random/uniform_int.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/gamma_distribution.hpp"
#include "boost/random/binomial_distribution.hpp"
#include "boost/random/poisson_distribution.hpp"
#include "boost/random/variate_generator.hpp"

#include "thrust/binary_search.h"

inline void bi::RngHost::seed(const unsigned seed) {
  rng.seed(seed);
}

template<class T1>
inline T1 bi::RngHost::uniformInt(const T1 lower, const T1 upper) {
  /* pre-condition */
  BI_ASSERT(upper >= lower);

  typedef boost::uniform_int<T1> dist_type;

  dist_type dist(lower, upper);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class V1>
inline typename V1::difference_type bi::RngHost::multinomial(const V1 lps) {
  /* pre-condition */
  BI_ASSERT(lps.size() > 0);

  typedef boost::uniform_real<typename V1::value_type> dist_type;

  typename sim_temp_vector<V1>::type Ps(lps.size());
  sumexpu_inclusive_scan(lps, Ps);

  typename V1::value_type sumexpu(*(Ps.end() - 1));
  if (sumexpu > 0) {
    dist_type dist(0.0, sumexpu);
    boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

    return thrust::lower_bound(Ps.begin(), Ps.end(), gen()) - Ps.begin();
  } else {
    return thrust::lower_bound(Ps.begin(), Ps.end(), .0) - Ps.begin();
  }
}

template<class T1>
inline T1 bi::RngHost::uniform(const T1 lower, const T1 upper) {
  /* pre-condition */
  BI_ASSERT(upper >= lower);

  typedef boost::uniform_real<T1> dist_type;

  dist_type dist(lower, upper);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class T1>
inline T1 bi::RngHost::gaussian(const T1 mu, const T1 sigma) {
  /* pre-condition */
  BI_ASSERT(sigma >= 0.0);

  typedef boost::normal_distribution<T1> dist_type;

  dist_type dist(mu, sigma);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return gen();
}

template<class T1>
inline T1 bi::RngHost::gamma(const T1 alpha, const T1 beta) {
  /* pre-condition */
  BI_ASSERT(alpha > 0.0 && beta > 0.0);

  typedef boost::gamma_distribution<T1> dist_type;

  dist_type dist(alpha);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return beta*gen();
}

template<class T1>
inline T1 bi::RngHost::poisson(const T1 lambda) {
  /* pre-condition */
  BI_ASSERT(lambda >= 0.0);

  if (lambda > 0) {
    typedef boost::poisson_distribution<int,T1> dist_type;

    dist_type dist(lambda);
    boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

    return static_cast<T1>(gen());
  } else {
    return 0;
  }

}

template<class T1, class T2>
inline T1 bi::RngHost::binomial(const T1 n, const T2 p) {
  /* pre-condition */
  BI_ASSERT(n >= static_cast<T1>(0.0) &&
            p >= static_cast<T2>(0.0) && p <= static_cast<T2>(1.0));

  typedef boost::binomial_distribution<int,T2> dist_type;

  dist_type dist(n, p);
  boost::variate_generator<rng_type&, dist_type> gen(rng, dist);

  return static_cast<T1>(gen());
}

#endif
