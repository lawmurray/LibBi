/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RANDOM_RANDOM_HPP
#define BI_RANDOM_RANDOM_HPP

#include "Rng.hpp"
#include "../misc/assert.hpp"
#include "../misc/location.hpp"
#include "../cuda/cuda.hpp"

namespace bi {
/**
 * Manager for pseudorandom number generation (PRNG).
 *
 * @ingroup math_rng
 *
 * Supports multithreaded (and thread safe) PRNG on both host and device,
 * using one PRNG per host thread, and one PRNG per device thread.
 *
 * The high-level interface of Random includes singular functions for
 * generating single variates (e.g. #uniform, #gaussian), plural functions
 * for filling vectors with variates (e.g. #uniforms, #gaussians), and
 * getters and setters for each thread's PRNG.
 *
 * More efficient than multiple calls to the singular functions is to use
 * #getHostRng or #getDevRng to get the base PRNG for the current thread,
 * and generate several variates from it.
 *
 * This is particularly the case with device generation, where, within a
 * kernel function, a variable of type <tt>Rng<ON_DEVICE></tt> can be
 * created in local memory, the PRNG for the current thread retrieve via
 * #getDevRng and copied into it, then variates generated from the local
 * variable before it is copied back to global memory with #setDevRng.
 *
 * Internally, the plural methods take this approach.
 */
class Random {
public:
  /**
   * Constructor. Initialise but do not seed random number generator.
   */
  Random();

  /**
   * Constructor. Initialise and seed all random number generators.
   *
   * @param seed Seed value.
   *
   * @seealso #seeds
   */
  Random(const unsigned seed);

  /**
   * Shallow copy constructor.
   */
  Random(const Random& o);

  /**
   * Destructor.
   */
  ~Random();

  /**
   * @name Singular methods
   */
  //@{
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
  //@}

  /**
   * @name Plural methods
   */
  //@{
  /**
   * Seed all random number generators.
   *
   * @param seed Seed value.
   *
   * All random number generators are seeded differently using a function of
   * @p seed.
   */
  void seeds(const unsigned seed);

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
  //@}

  /**
   * @name Base PRNGs
   */
  //@{
  /**
   * Get the current host thread's random number generator.
   */
  Rng<ON_HOST>& getHostRng();

  #ifdef ENABLE_GPU
  /**
   * Get the current device thread's random number generator.
   */
  CUDA_FUNC_DEVICE Rng<ON_DEVICE>& getDevRng();

  /**
   * Set the current device thread's random number generator. This is
   * typically used when the state of the random number generator is read
   * into local memory and modified, and so must be copied back to global
   * memory.
   */
  CUDA_FUNC_DEVICE void setDevRng(const Rng<ON_DEVICE>& rng);
  #endif
  //@}

private:
  /**
   * Seed random number generators on host.
   */
  void hostSeeds(const unsigned seed);

  /**
   * Seed random number generators on device.
   */
  void devSeeds(const unsigned seed);

  /**
   * Random number generators on host.
   */
  Rng<ON_HOST>* hostRngs;

  #ifdef ENABLE_GPU
  /**
   * Random number generators on device.
   */
  Rng<ON_DEVICE>* devRngs;
  #endif

  /**
   * Does this object own the host and device random number generators? This
   * is largely used to ensure that when copying to device in a kernel
   * launch, the random number generators are not destroyed on exit.
   */
  bool own;
};
}

inline void bi::Random::seed(const unsigned seed) {
  getHostRng().seed(seed);
}

template<class T1>
inline T1 bi::Random::uniformInt(const T1 lower, const T1 upper) {
  return getHostRng().uniformInt(lower, upper);
}

template<class V1>
inline typename V1::difference_type bi::Random::multinomial(const V1 ps) {
  return getHostRng().multinomial(ps);
}

template<class T1>
inline T1 bi::Random::uniform(const T1 lower, const T1 upper) {
  return getHostRng().uniform(lower, upper);
}

template<class T1>
inline T1 bi::Random::gaussian(const T1 mu, const T1 sigma) {
  return getHostRng().gaussian(mu, sigma);
}

template<class T1>
inline T1 bi::Random::gamma(const T1 alpha, const T1 beta) {
  return getHostRng().gamma(alpha, beta);
}

template<class V1>
void bi::Random::uniforms(V1 x, const typename V1::value_type lower,
    const typename V1::value_type upper) {
  /* pre-condition */
  assert (upper >= lower);

  typedef typename V1::value_type T1;
  typedef boost::uniform_real<T1> dist_type;

  if (V1::on_device) {
    BI_ERROR(false, "Not implemented");
  } else {
    #pragma omp parallel
    {
      BOOST_AUTO(rng, getHostRng());
      int j;

      #pragma omp for schedule(static)
      for (j = 0; j < x.size(); ++j) {
        x(j) = rng.uniform(lower, upper);
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
  typedef boost::normal_distribution<T1> dist_type;

  if (V1::on_device) {
    BI_ERROR(false, "Not implemented");
  } else {
    #pragma omp parallel
    {
      BOOST_AUTO(rng, getHostRng());
      int j;

      #pragma omp for schedule(static)
      for (j = 0; j < x.size(); ++j) {
        x(j) = rng.gaussian(mu, sigma);
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
  typedef boost::gamma_distribution<T1> dist_type;

  if (V1::on_device) {
    BI_ERROR(false, "Not implemented");
  } else {
    #pragma omp parallel
    {
      BOOST_AUTO(rng, getHostRng());
      int j;

      #pragma omp for schedule(static)
      for (j = 0; j < x.size(); ++j) {
        x(j) = rng.gamma(alpha, beta);
      }
    }
  }
}

template<class V1, class V2>
void bi::Random::multinomials(const V1 ps, V2 xs) {
  /* pre-condition */
  assert (ps.size() > 0);

  typedef typename V1::value_type T1;

  if (V1::on_device) {
    BI_ERROR(false, "Not implemented");
  } else {
    BOOST_AUTO(rng, getHostRng());
    typename sim_temp_vector<V1>::type Ps(ps.size());
    inclusive_scan_sum_exp(ps, Ps);

    T1 u;
    T1 lower = 0.0;
    T1 upper = *(Ps.end() - 1);

    int i, p;
    for (i = 0; i < xs.size(); ++i) {
      u = rng.uniform(lower, upper);
      p = thrust::lower_bound(Ps.begin(), Ps.end(), u) - Ps.begin();

      xs(i) = p;
    }
  }
}

inline bi::Rng<bi::ON_HOST>& bi::Random::getHostRng() {
  return hostRngs[bi_omp_tid];
}

#ifdef __CUDACC__
#include "../cuda/random/Random.cuh"
#endif

#endif
