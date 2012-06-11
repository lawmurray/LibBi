/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 2589 $
 * $Date: 2012-05-23 13:15:11 +0800 (Wed, 23 May 2012) $
 */
#ifndef BI_RANDOM_RNG_HPP
#define BI_RANDOM_RNG_HPP

#include "../cuda/cuda.hpp"
#include "../misc/location.hpp"

#include "boost/random/mersenne_twister.hpp"

#ifdef ENABLE_GPU
#include "curand_kernel.h"
#endif

namespace bi {
/**
 * Pseudorandom number generator.
 *
 * @ingroup math_rng
 *
 * @tparam L Location.
 */
template<Location L>
class Rng {
  //
};

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
template<>
class Rng<ON_HOST> {
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

#ifdef ENABLE_GPU
/**
 * Pseudorandom number generator, on device.
 *
 * @ingroup math_rng
 */
template<>
class Rng<ON_DEVICE> {
public:
  /**
   * @copydoc Rng<ON_HOST>::seed
   */
  CUDA_FUNC_DEVICE void seed(const unsigned seed);

  /**
   * @copydoc Rng<ON_HOST>::uniformInt
   */
  template<class T1>
  CUDA_FUNC_DEVICE T1 uniformInt(const T1 lower = 0, const T1 upper = 1);

  /**
   * @copydoc Rng<ON_HOST>::uniform
   */
  CUDA_FUNC_DEVICE float uniform(const float lower = 0.0f, const float upper = 1.0f);

  /**
   * @copydoc Rng<ON_HOST>::uniform
   */
  CUDA_FUNC_DEVICE double uniform(const double lower = 0.0, const double upper = 1.0);

  /**
   * @copydoc Rng<ON_HOST>::gaussian
   */
  CUDA_FUNC_DEVICE float gaussian(const float mu = 0.0f, const float sigma = 1.0f);

  /**
   * @copydoc Rng<ON_HOST>::gaussian
   */
  CUDA_FUNC_DEVICE double gaussian(const double mu = 0.0, const double sigma = 1.0);

private:
  /**
   * CURAND state.
   */
  curandState rng;
};
#endif

}

#include "../host/random/Rng.hpp"
#ifdef __CUDACC__
#include "../cuda/random/Rng.cuh"
#endif

#endif
