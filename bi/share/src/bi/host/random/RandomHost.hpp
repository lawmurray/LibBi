/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RANDOM_RANDOMHOST_HPP
#define BI_RANDOM_RANDOMHOST_HPP

#include "../../misc/assert.hpp"
#include "../../misc/location.hpp"
#include "../../cuda/cuda.hpp"

namespace bi {
class Random;

/**
 * Implementation of Random on host.
 */
struct RandomHost {
  /**
   * @copydoc Random::seeds
   */
  static void seeds(Random& rng, const unsigned seed);

  /**
   * @copydoc Random::uniforms
   */
  template<class V1>
  static void uniforms(Random& rng, V1 x,
      const typename V1::value_type lower = 0.0,
      const typename V1::value_type upper = 1.0);

  /**
   * @copydoc Random::gaussians
   */
  template<class V1>
  static void gaussians(Random& rng, V1 x, const typename V1::value_type mu =
      0.0, const typename V1::value_type sigma = 1.0);

  /**
   * @copydoc Random::gammas
   */
  template<class V1>
  static void gammas(Random& rng, V1 x, const typename V1::value_type alpha =
      1.0, const typename V1::value_type beta = 1.0);

  /**
   * @copydoc Random::multinomials
   */
  template<class V1, class V2>
  static void multinomials(Random& rng, const V1 ps, V2 xs);

};
}

#include "../../random/Random.hpp"

template<class V1>
void bi::RandomHost::uniforms(Random& rng, V1 x,
    const typename V1::value_type lower,
    const typename V1::value_type upper) {
  /* pre-condition */
  assert(upper >= lower);

  typedef typename V1::value_type T1;
  typedef boost::uniform_real<T1> dist_type;

  #pragma omp parallel
  {
    BOOST_AUTO(rng1, rng.getHostRng());
    int j;

    #pragma omp for schedule(static)
    for (j = 0; j < x.size(); ++j) {
      x(j) = rng1.uniform(lower, upper);
    }
  }
}

template<class V1>
void bi::RandomHost::gaussians(Random& rng, V1 x,
    const typename V1::value_type mu, const typename V1::value_type sigma) {
  /* pre-condition */
  assert(sigma >= 0.0);

  typedef typename V1::value_type T1;
  typedef boost::normal_distribution<T1> dist_type;

  #pragma omp parallel
  {
    BOOST_AUTO(rng1, rng.getHostRng());
    int j;

    #pragma omp for schedule(static)
    for (j = 0; j < x.size(); ++j) {
      x(j) = rng1.gaussian(mu, sigma);
    }
  }
}

template<class V1>
void bi::RandomHost::gammas(Random& rng, V1 x,
    const typename V1::value_type alpha, const typename V1::value_type beta) {
  /* pre-condition */
  assert(alpha > 0.0 && beta > 0.0);

  typedef typename V1::value_type T1;
  typedef boost::gamma_distribution<T1> dist_type;

  #pragma omp parallel
  {
    BOOST_AUTO(rng1, rng.getHostRng());
    int j;

    #pragma omp for schedule(static)
    for (j = 0; j < x.size(); ++j) {
      x(j) = rng1.gamma(alpha, beta);
    }
  }
}

template<class V1, class V2>
void bi::RandomHost::multinomials(Random& rng, const V1 ps, V2 xs) {
  /* pre-condition */
  assert(ps.size() > 0);

  typedef typename V1::value_type T1;

  BOOST_AUTO(rng1, rng.getHostRng());
  typename sim_temp_vector<V1>::type Ps(ps.size());
  inclusive_scan_sum_exp(ps, Ps);

  T1 u;
  T1 lower = 0.0;
  T1 upper = *(Ps.end() - 1);

  int i, p;
  for (i = 0; i < xs.size(); ++i) {
    u = rng1.uniform(lower, upper);
    p = thrust::lower_bound(Ps.begin(), Ps.end(), u) - Ps.begin();

    xs(i) = p;
  }
}

#endif
