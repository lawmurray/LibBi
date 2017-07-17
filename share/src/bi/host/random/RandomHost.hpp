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
   * @copydoc Random::poissons
   */
  template<class V1>
  static void poissons(Random& rng, V1 x, const typename V1::value_type lambda =
                     1.0);

  /**
   * @copydoc Random::binomials
   */
  template<class V1, class V2>
  static void binomials(Random& rng, V1 x, const typename V1::value_type n =
                        1.0, const typename V2::value_type p = 0.5);

  /**
   * @copydoc Random::betas
   */
  template<class V1>
  static void betas(Random& rng, V1 x, const typename V1::value_type alpha =
      1.0, const typename V1::value_type beta = 1.0);

  /**
   * @copydoc Random::multinomials
   */
  template<class V1, class V2>
  static void multinomials(Random& rng, const V1 lps, V2 xs);
};
}

#include "../../random/Random.hpp"

template<class V1>
void bi::RandomHost::uniforms(Random& rng, V1 x,
    const typename V1::value_type lower,
    const typename V1::value_type upper) {
  /* pre-condition */
  BI_ASSERT(upper >= lower);

  typedef typename V1::value_type T1;
  typedef boost::uniform_real<T1> dist_type;

  //#pragma omp parallel
  //{
    RngHost& rng1 = rng.getHostRng();
    int j;

    dist_type dist(lower, upper);
    boost::variate_generator<RngHost::rng_type&, dist_type> gen(rng1.rng, dist);

    //#pragma omp for
    for (j = 0; j < x.size(); ++j) {
      x(j) = gen();
    }
    //}
}

template<class V1>
void bi::RandomHost::gaussians(Random& rng, V1 x,
    const typename V1::value_type mu, const typename V1::value_type sigma) {
  /* pre-condition */
  BI_ASSERT(sigma >= 0.0);

  typedef typename V1::value_type T1;
  typedef boost::normal_distribution<T1> dist_type;

  //#pragma omp parallel
  //{
    RngHost& rng1 = rng.getHostRng();
    int j;

    dist_type dist(mu, sigma);
    boost::variate_generator<RngHost::rng_type&, dist_type> gen(rng1.rng, dist);

    //#pragma omp for schedule(static)
    for (j = 0; j < x.size(); ++j) {
      x(j) = gen();
    }
  //}
}

template<class V1>
void bi::RandomHost::gammas(Random& rng, V1 x,
    const typename V1::value_type alpha, const typename V1::value_type beta) {
  /* pre-condition */
  BI_ASSERT(alpha > 0.0 && beta > 0.0);

  typedef typename V1::value_type T1;
  typedef boost::gamma_distribution<T1> dist_type;

  //#pragma omp parallel
  //{
    RngHost& rng1 = rng.getHostRng();
    int j;

    dist_type dist(alpha);
    boost::variate_generator<RngHost::rng_type&, dist_type> gen(rng1.rng, dist);

    //#pragma omp for schedule(static)
    for (j = 0; j < x.size(); ++j) {
      x(j) = beta*gen();
    }
    //}
}

template<class V1>
void bi::RandomHost::betas(Random& rng, V1 x,
    const typename V1::value_type alpha, const typename V1::value_type beta) {
  /* pre-condition */
  BI_ASSERT(alpha > 0.0 && beta > 0.0);

  typedef typename V1::value_type T1;
  typedef boost::gamma_distribution<T1> dist_type;

  //#pragma omp parallel
  //{
    RngHost& rng1 = rng.getHostRng();
    int j;

    dist_type dist1(alpha), dist2(beta);
    boost::variate_generator<RngHost::rng_type&, dist_type> gen1(rng1.rng, dist1), gen2(rng1.rng, dist2);
    T1 y1, y2;

    //#pragma omp for schedule(static)
    for (j = 0; j < x.size(); ++j) {
      y1 = gen1();
      y2 = gen2();

      x(j) = y1/(y1 + y2);
    }
    //}
}

template<class V1, class V2>
void bi::RandomHost::multinomials(Random& rng, const V1 lps, V2 xs) {
  /* pre-condition */
  BI_ASSERT(lps.size() > 0);

  typedef typename V1::value_type T1;

  RngHost& rng1 = rng.getHostRng();
  typename sim_temp_vector<V1>::type Ps(lps.size());
  sumexpu_inclusive_scan(lps, Ps);

  T1 u;
  T1 lower = 0.0;
  T1 upper = *(Ps.end() - 1);

  int i, p;
  for (i = 0; i < xs.size(); ++i) {
    ///@todo Review implementation, in particular the next two repeated calls
    u = rng1.uniform(lower, upper);
    p = thrust::lower_bound(Ps.begin(), Ps.end(), u) - Ps.begin();

    xs(i) = p;
  }
}

#endif
