/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Imported from dysii 1.4.0, originally indii/ml/aux/Random.hpp
 */
#ifndef BI_RANDOM_RANDOM_HPP
#define BI_RANDOM_RANDOM_HPP

#include "../cuda/cuda.hpp"
#include "../cuda/random/curand.hpp"
#include "../misc/assert.hpp"

#include "boost/random/mersenne_twister.hpp"

#include <vector>

namespace bi {
/**
 * Pseudorandom number generator.
 *
 * @ingroup math_rng
 *
 * Uses the Mersenne Twister algorithm for generating pseudorandom variates,
 * as implemented by Boost.
 *
 * The functions of this class are thread-safe (using OpenMP), utilising one
 * random number generator per thread. Each random number generator is
 * seeded differently from the one seed given. Experimental results suggest
 * that this is sufficient with the Mersenne Twister given its large period.
 * Parallel generation of random numbers is used in the plural methods,
 * with static scheduling to ensure reproducibility with the same seed.
 *
 * @section Random_references References
 *
 * @anchor Matsumoto1998 Matsumoto, M. and Nishimura,
 * T. Mersenne Twister: A 623-dimensionally equidistributed
 * uniform pseudorandom number generator. <i>ACM Transactions on
 * Modeling and Computer Simulation</i>, <b>1998</b>, 8, 3-30.
 */
class Random {
public:
  /**
   * Constructor. Initialise but do not seed random number generator.
   */
  Random();

  /**
   * Constructor. Initialise and seed random number generator.
   *
   * @param seed Seed value.
   */
  Random(const unsigned seed);

  /**
   * Destructor.
   */
  ~Random();

  /**
   * Seed random number generator.
   *
   * @param seed Seed value.
   */
  void seed(const unsigned seed);

  /**
   * Reset random number generator with last used seed.
   */
  void reset();

  /**
   * Generate a boolean value from a Bernoulli distribution.
   *
   * @param p Probability of true, between 0 and 1 inclusive.
   *
   * @return The random boolean value, 1 for true, 0 for false.
   */
  template<class T1>
  unsigned bernoulli(const T1 p = 0.5);

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

  /**
   * Generate random numbers from a multinomial distribution with given
   * probabilities.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Vector type.
   *
   * @param ps Log-probabilities. Need not be normalised.
   * @param[out] xs Random indices between @c 0 and <tt>ps.size() - 1</tt>,
   * selected according to the non-normalised probabilities given in @c ps.
   */
  template<class V1, class V2>
  void multinomials(const V1 ps, V2 xs);

  /**
   * Fill vector with random numbers from a uniform distribution over a
   * given interval.
   *
   * @tparam V1 Vector type.
   *
   * @param lower Lower bound on the interval.
   * @param upper Upper bound on the interval.
   * @param[out] x Vector.
   */
  template<class V1>
  void uniforms(V1 x, const typename V1::value_type lower = 0.0,
      const typename V1::value_type upper = 1.0);

  /**
   * Fill vector with random numbers from a Gaussian distribution with a
   * given mean and standard deviation.
   *
   * @tparam V1 Vector type.
   *
   * @param mu Mean of the distribution.
   * @param sigma Standard deviation of the distribution.
   * @param[out] x Vector.
   */
  template<class V1>
  void gaussians(V1 x, const typename V1::value_type mu = 0.0,
      const typename V1::value_type sigma = 1.0);

  /**
   * Fill vector with random numbers from a gamma distribution with a given
   * shape and scale.
   *
   * @tparam T1 Scalar type.
   *
   * @param alpha Shape.
   * @param beta Scale.
   *
   * @param[out] x Vector.
   */
  template<class V1>
  void gammas(V1 x, const typename V1::value_type alpha = 1.0,
      const typename V1::value_type beta = 1.0);

private:
  /**
   * Type of random number generator on host.
   */
  typedef boost::mt19937 rng_t;

  /**
   * Random number generators on host.
   */
  std::vector<rng_t> rng;

  /**
   * Random number generator on device.
   */
  #ifndef USE_CPU
  curandGenerator_t devRng;
  #endif

  /**
   * Original seed.
   */
  unsigned originalSeed;
};

}

#include "../misc/omp.hpp"
#include "../math/temp_vector.hpp"
#include "../cuda/math/temp_vector.hpp"

#include "boost/random/uniform_int.hpp"
#include "boost/random/bernoulli_distribution.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/normal_distribution.hpp"
#include "boost/random/gamma_distribution.hpp"
#include "boost/random/variate_generator.hpp"

#include "thrust/binary_search.h"

template<class T1>
inline unsigned bi::Random::bernoulli(const T1 p) {
  /* pre-condition */
  assert (p >= 0.0 && p <= 1.0);

  typedef boost::bernoulli_distribution<unsigned> dist_t;

  dist_t dist(p);
  boost::variate_generator<rng_t&, dist_t> gen(rng[bi_omp_tid], dist);

  return gen();
}

template<class T1>
inline T1 bi::Random::uniformInt(const T1 lower, const T1 upper) {
  /* pre-condition */
  assert (upper >= lower);

  typedef boost::uniform_int<T1> dist_t;

  dist_t dist(lower, upper);
  boost::variate_generator<rng_t&, dist_t> gen(rng[bi_omp_tid], dist);

  return gen();
}

template<class V1>
inline typename V1::difference_type bi::Random::multinomial(const V1 ps) {
  /* pre-condition */
  assert (ps.size() > 0);

  typedef boost::uniform_real<typename V1::value_type> dist_t;

  BOOST_AUTO(Ps, temp_vector<V1>(ps.size()));
  exclusive_scan_sum_exp(ps.begin(), ps.end(), Ps->begin());

  dist_t dist(0, *(Ps->end() - 1));
  boost::variate_generator<rng_t&, dist_t> gen(rng[bi_omp_tid], dist);

  int p = thrust::lower_bound(Ps->begin(), Ps->end(), gen()) - Ps->begin();
  delete Ps;

  return p;
}

template<class T1>
inline T1 bi::Random::uniform(const T1 lower, const T1 upper) {
  /* pre-condition */
  assert (upper >= lower);

  typedef boost::uniform_real<T1> dist_t;

  dist_t dist(lower, upper);
  boost::variate_generator<rng_t&, dist_t> gen(rng[bi_omp_tid], dist);

  return gen();
}

template<class T1>
inline T1 bi::Random::gaussian(const T1 mu, const T1 sigma) {
  /* pre-condition */
  assert (sigma >= 0.0);

  typedef boost::normal_distribution<T1> dist_t;

  dist_t dist(mu, sigma);
  boost::variate_generator<rng_t&, dist_t> gen(rng[bi_omp_tid], dist);

  return gen();
}

template<class T1>
inline T1 bi::Random::gamma(const T1 alpha, const T1 beta) {
  /* pre-condition */
  assert (alpha > 0.0 && beta > 0.0);

  typedef boost::gamma_distribution<T1> dist_t;

  dist_t dist(alpha);
  boost::variate_generator<rng_t&, dist_t> gen(rng[bi_omp_tid], dist);

  return beta*gen();
}

template<class V1>
void bi::Random::uniforms(V1 x, const typename V1::value_type lower,
    const typename V1::value_type  upper) {
  /* pre-condition */
  assert (upper >= lower);

  typedef typename V1::value_type T1;
  typedef boost::uniform_real<T1> dist_t;

  if (V1::on_device) {
    assert (lower == 0.0 && upper == 1.0);
    #ifndef USE_CPU
    CURAND_CHECKED_CALL(curand_generate_uniform<T1>::func(devRng, x.buf(),
        x.size()));
    #else
    BI_ASSERT(false, "GPU random number generation not enabled");
    #endif
  } else {
    #pragma omp parallel
    {
      int j;
      dist_t dist(lower, upper);
      boost::variate_generator<rng_t&,dist_t> gen(rng[bi_omp_tid], dist);

      #pragma omp for schedule(static)
      for (j = 0; j < x.size(); ++j) {
        x(j) = gen();
      }
    }
  }
}

template<class V1>
void bi::Random::gaussians(V1 x, const typename V1::value_type mu,
    const typename V1::value_type sigma) {
  /* pre-condition */
  assert (sigma >= 0.0);

  typedef typename V1::value_type T1;
  typedef boost::normal_distribution<T1> dist_t;

  if (V1::on_device) {
    #ifndef USE_CPU
    CURAND_CHECKED_CALL(curand_generate_normal<T1>::func(devRng, x.buf(),
        x.size(), static_cast<T1>(0.0), static_cast<T1>(1.0)));
    #else
    BI_ASSERT(false, "GPU random number generation not enabled");
    #endif
  } else {
    #pragma omp parallel
    {
      int j;
      dist_t dist(mu, sigma);
      boost::variate_generator<rng_t&,dist_t> gen(rng[bi_omp_tid], dist);

      #pragma omp for schedule(static)
      for (j = 0; j < x.size(); ++j) {
        x(j) = gen();
      }
    }
  }
}

template<class V1>
void bi::Random::gammas(V1 x, const typename V1::value_type alpha,
    const typename V1::value_type beta) {
  /* pre-condition */
  assert (alpha > 0.0 && beta > 0.0);

  typedef typename V1::value_type T1;
  typedef boost::gamma_distribution<T1> dist_t;

  int j;
  dist_t dist(alpha);
  boost::variate_generator<rng_t&,dist_t> gen(rng[bi_omp_tid], dist);

  if (V1::on_device) {
    /* CURAND doesn't support gamma variates at this stage, generate on host
     * and upload */
    BOOST_AUTO(y, host_temp_vector<T1>(x.size()));
    BOOST_AUTO(z, *y);
    #pragma omp parallel for schedule(static)
    for (j = 0; j < x.size(); ++j) {
      z(j) = gen();
    }
    x = z;
    synchronize();
    delete y;
  } else {
    #pragma omp parallel for schedule(static)
    for (j = 0; j < x.size(); ++j) {
      x(j) = gen();
    }
  }
  scal(beta, x);
}

template<class V1, class V2>
void bi::Random::multinomials(const V1 ps, V2 xs) {
  /* pre-condition */
  assert (ps.size() > 0);

  typedef boost::uniform_real<typename V1::value_type> dist_t;

  BOOST_AUTO(Ps, temp_vector<V1>(ps.size()));
  exclusive_scan_sum_exp(ps.begin(), ps.end(), Ps->begin());

  dist_t dist(0, *(Ps->end() - 1));
  boost::variate_generator<rng_t&, dist_t> gen(rng[bi_omp_tid], dist);

  int i, p;
  for (i = 0; i < xs.size(); ++i) {
    p = thrust::lower_bound(Ps->begin(), Ps->end(), gen()) - Ps->begin();
    xs(i) = p;
  }

  delete Ps;
}

#endif
