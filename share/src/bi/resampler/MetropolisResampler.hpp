/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_METROPOLISRESAMPLER_HPP
#define BI_RESAMPLER_METROPOLISRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * MetropolisResampler implementation on host.
 */
class MetropolisResamplerHost: public ResamplerBaseHost {
public:
  /**
   * @copydoc MetropolisResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as, int B);

  /**
   * @copydoc MetropolisResampler::ancestorsPermute()
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as, int B);
};

/**
 * MetropolisResampler implementation on device.
 */
class MetropolisResamplerGPU: public ResamplerBaseGPU {
public:
  /**
   * @copydoc MetropolisResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as, int B);

  /**
   * @copydoc MetropolisResampler::ancestorsPermute()
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as, int B);
};

/**
 * Metropolis resampler for particle filter.
 *
 * @ingroup method_resampler
 *
 * Implements the Metropolis resampler as described in @ref Murray2011a
 * "Murray (2011)" and @ref Murray2014 "Murray, Lee & Jacob (2014)".
 */
class MetropolisResampler: public ResamplerBase {
public:
  /**
   * Constructor.
   *
   * @param B Number of Metropolis steps to take.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param bridgeEssRel Minimum ESS, as proportion of total number of
   * particles, to trigger resampling after bridge weighting.
   */
  MetropolisResampler(const int B, const double essRel = 0.5,
      const double bridgeEssRel = 0.5);

  /**
   * Set number of steps to take.
   */
  void setSteps(const int B);

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc MultinomialResampler::ancestors
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as,
      ResamplerBasePrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc MultinomialResampler::ancestorsPermute
   */
  template<class V1, class V2>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      ResamplerBasePrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);
  //@}

private:
  /**
   * Number of Metropolis steps to take.
   */
  int B;
};

/**
 * @internal
 */
template<Location L>
struct precompute_type<MetropolisResampler,L> {
  typedef ResamplerBasePrecompute<L> type;
};
}

#include "../host/resampler/MetropolisResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/MetropolisResamplerGPU.cuh"
#endif

template<class V1, class V2, bi::Location L>
void bi::MetropolisResampler::ancestors(Random& rng, const V1 lws, V2 as,
    ResamplerBasePrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,MetropolisResamplerGPU,
      MetropolisResamplerHost>::type impl;
  impl::ancestors(rng, lws, as, B);
}

template<class V1, class V2, bi::Location L>
void bi::MetropolisResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, ResamplerBasePrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,MetropolisResamplerGPU,
      MetropolisResamplerHost>::type impl;
  impl::ancestorsPermute(rng, lws, as, B);
}

#endif
