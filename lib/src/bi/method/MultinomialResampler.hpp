/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MULTINOMIALRESAMPLER_HPP
#define BI_METHOD_MULTINOMIALRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"

namespace bi {
/**
 * Multinomial resampler for particle filter.
 *
 * @ingroup method
 */
class MultinomialResampler : public Resampler {
public:
  /**
   * Constructor.
   *
   * @param rng Random number generator.
   * @param sort True to pre-sort weights, false otherwise.
   */
  MultinomialResampler(Random& rng, const bool sort = true);

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
   * Pre-sort weights?
   */
  bool sort;
};

}

#include "MultinomialResampler.inl"

#endif
