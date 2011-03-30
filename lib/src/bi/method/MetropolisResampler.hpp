/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_METROPOLISRESAMPLER_HPP
#define BI_METHOD_METROPOLISRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"

namespace bi {
/**
 * @internal
 *
 * MetropolisResampler implementation on device.
 */
class MetropolisResamplerDeviceImpl {
public:
  /**
   * @copydoc MetropolisResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(const V1& lws, V2& as, Random& rng, int L);
};

/**
 * @internal
 *
 * MetropolisResampler implementation on host.
 */
class MetropolisResamplerHostImpl {
public:
  /**
   * @copydoc MetropolisResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(const V1& lws, V2& as, Random& rng, int L);
};

/**
 * Metropolis resampler for particle filter.
 *
 * @ingroup method
 *
 * Implements the Metropolis resampler as described in @ref Murray2011a
 * "Murray (2011)".
 */
class MetropolisResampler : public Resampler {
public:
  /**
   * Constructor.
   *
   * @param rng A random number generator.
   * @param L Number of Metropolis steps to take.
   * @param A Number of accelerated steps to take.
   */
  MetropolisResampler(Random& rng, const int L, const int A = 0);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(V1&, V2&)
   */
  template<class V1, class V2>
  void resample(V1& lws, V2& as);

  /**
   * @copydoc concept::Resampler::resample(const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3>
  void resample(const V1& qlws, V2& lws, V3& as);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, V1&, V2&)
   */
  template<class V1, class V2>
  void resample(const int a, V1& lws, V2& as);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3>
  void resample(const int a, const V1& qlws, V2& lws, V3& as);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Select ancestors.
   *
   * @tparam V1 Floating point vector type.
   * @tparam V2 Integer vector type.
   *
   * @param lws Log-weights.
   * @param[out] as Ancestry.
   */
  template<class V1, class V2>
  void ancestors(const V1& lws, V2& as);
  //@}

private:
  /**
   * %Random number generator.
   */
  Random& rng;

  /**
   * Number of Metropolis steps to take.
   */
  int L;

  /**
   * Number of accelerated steps to take.
   */
  int A;
};

}

#include "MetropolisResampler.inl"

#endif
