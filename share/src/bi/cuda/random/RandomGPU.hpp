/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RANDOM_RANDOMGPU_HPP
#define BI_RANDOM_RANDOMGPU_HPP

namespace bi {
class Random;

/**
 * Implementation of Random on device.
 */
struct RandomGPU {
  /**
   * @copydoc Random::seeds
   */
  static void seeds(Random& rng, const unsigned seed);

  /**
   * @copydoc Random::multinomials
   */
  template<class V1, class V2>
  static void multinomials(Random& rng, const V1 lps, V2 xs);

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
  static void poissons(Random& rng, V1 x, const typename V1::value_type labmda =
                     1.0);

  /**
   * @copydoc Random::binomial
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
};
}

#ifdef __CUDACC__
#include "RandomGPU.cuh"
#endif

#endif
