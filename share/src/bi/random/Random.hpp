/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RANDOM_RANDOM_HPP
#define BI_RANDOM_RANDOM_HPP

#include "../host/random/RngHost.hpp"
#include "../misc/assert.hpp"
#include "../misc/location.hpp"
#include "../cuda/cuda.hpp"

#ifdef ENABLE_CUDA
#include "../cuda/random/curandStateSA.hpp"
#endif

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
 * kernel function, a variable of type <tt>RngGPU</tt> can be
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
   * @param lps Log-probabilities. Need not be normalised.
   *
   * @return Random index between @c 0 and <tt>ps.size() - 1</tt>, selected
   * according to the non-normalised log-probabilities given in @c ps.
   */
  template<class V1>
  typename V1::difference_type multinomial(const V1 lps);

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
   * @param lps Log-probabilities. Need not be normalised.
   * @param[out] xs Random indices between @c 0 and <tt>ps.size() - 1</tt>,
   * selected according to the non-normalised log-probabilities given in
   * @c lps.
   */
  template<class V1, class V2>
  void multinomials(const V1 lps, V2 xs);

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

  /**
   * Fill vector with random numbers from a beta distribution with given
   * parameters.
   *
   * @tparam T1 Scalar type.
   *
   * @param alpha Shape.
   * @param beta Shape.
   *
   * @param[out] x Vector.
   */
  template<class V1>
  void betas(V1 x, const typename V1::value_type alpha = 1.0,
      const typename V1::value_type beta = 1.0);
  //@}

  /**
   * @name Base PRNGs
   */
  //@{
  /**
   * Get the current host thread's random number generator.
   */
  RngHost& getHostRng();

#ifdef ENABLE_CUDA
  /**
   * Get a thread's random number generator.
   *
   * @param p Thread number.
   */
  //CUDA_FUNC_DEVICE curandState& getDevRng(const int p);
#endif
  //@}

public:
  /**
   * Random number generators on host.
   */
  RngHost* hostRngs;

#ifdef ENABLE_CUDA
  /**
   * Random number generators on device.
   */
  curandStateSA devRngs;
#endif

  /**
   * Does this object own the host and device random number generators? This
   * is largely used to ensure that when copying to device in a kernel
   * launch, the random number generators are not destroyed on exit.
   */
  bool own;
};
}

#include "../host/random/RandomHost.hpp"
#ifdef ENABLE_CUDA
#include "../cuda/random/RandomGPU.hpp"
#endif

inline void bi::Random::seed(const unsigned seed) {
  getHostRng().seed(seed);
}

template<class T1>
inline T1 bi::Random::uniformInt(const T1 lower, const T1 upper) {
  return getHostRng().uniformInt(lower, upper);
}

template<class V1>
inline typename V1::difference_type bi::Random::multinomial(const V1 lps) {
  return getHostRng().multinomial(lps);
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
#ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<V1::on_device,RandomGPU,RandomHost>::type impl;
#else
  typedef RandomHost impl;
#endif
  impl::uniforms(*this, x, lower, upper);
}

template<class V1>
void bi::Random::gaussians(V1 x, const typename V1::value_type mu,
    const typename V1::value_type sigma) {
#ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<V1::on_device,RandomGPU,RandomHost>::type impl;
#else
  typedef RandomHost impl;
#endif
  impl::gaussians(*this, x, mu, sigma);
}

template<class V1>
void bi::Random::gammas(V1 x, const typename V1::value_type alpha,
    const typename V1::value_type beta) {
#ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<V1::on_device,RandomGPU,RandomHost>::type impl;
#else
  typedef RandomHost impl;
#endif
  impl::gammas(*this, x, alpha, beta);
}

template<class V1>
void bi::Random::betas(V1 x, const typename V1::value_type alpha,
    const typename V1::value_type beta) {
#ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<V1::on_device,RandomGPU,RandomHost>::type impl;
#else
  typedef RandomHost impl;
#endif
  impl::betas(*this, x, alpha, beta);
}

template<class V1, class V2>
void bi::Random::multinomials(const V1 lps, V2 xs) {
#ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<V1::on_device,RandomGPU,RandomHost>::type impl;
#else
  typedef RandomHost impl;
#endif
  impl::multinomials(*this, lps, xs);
}

inline bi::RngHost& bi::Random::getHostRng() {
  return hostRngs[bi_omp_tid];
}

#ifdef ENABLE_CUDA
//inline curandState& bi::Random::getDevRng(const int p) {
//  return devRngs[p];
//}
#endif

#endif
