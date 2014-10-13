/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_MULTINOMIALRESAMPLER_HPP
#define BI_RESAMPLER_MULTINOMIALRESAMPLER_HPP

#include "ScanResamplerBase.hpp"
#include "../random/Random.hpp"
#include "../cuda/cuda.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Multinomial resampler for particle filter.
 *
 * @ingroup method_resampler
 */
class MultinomialResampler: public ScanResamplerBase {
public:
  /**
   * Constructor.
   *
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param bridgeEssRel Minimum ESS, as proportion of total number of
   * particles, to trigger resampling after bridge weighting.
   */
  MultinomialResampler(const double essRel = 0.5, const double bridgeEssRel =
      0.5);

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Select ancestors.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param[out] as Ancestors.
   */
  template<class V1, class V2, Location L>
  void ancestors(Random& rng, const V1 lws, V2 as,
      ScanResamplerBasePrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * Select ancestors and permute.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param[out] as Ancestors.
   */
  template<class V1, class V2, Location L>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      ScanResamplerBasePrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);
//@}
};

/**
 * @internal
 */
template<Location L>
struct precompute_type<MultinomialResampler,L> {
  typedef ScanResamplerBasePrecompute<L> type;
};

}

#include "../host/resampler/MultinomialResamplerHost.hpp"
#ifdef ENABLE_CUDA
#include "../cuda/resampler/MultinomialResamplerGPU.hpp"
#endif

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as,
    ScanResamplerBasePrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
#ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<L,MultinomialResamplerGPU,
  MultinomialResamplerHost>::type impl;
#else
  typedef MultinomialResamplerHost impl;
#endif
  impl::ancestors(rng, lws, as, pre);
}

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, ScanResamplerBasePrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  ancestors(rng, lws, as, pre);
  permute(as);
}

#endif
