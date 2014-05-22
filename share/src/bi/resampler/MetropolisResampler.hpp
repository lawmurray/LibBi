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
class MetropolisResamplerHost: public ResamplerHost {
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
class MetropolisResamplerGPU: public ResamplerGPU {
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
class MetropolisResampler: public Resampler {
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
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(Random&, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  void resample(Random& rng, V1 lws, V2 as, O1& s);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::ancestors()
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::ancestors()
   */
  template<class V1, class V2>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::offspring()
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 os, const int P)
      throw (ParticleFilterDegeneratedException);
  //@}

private:
  /**
   * Number of Metropolis steps to take.
   */
  int B;
};
}

#include "../host/resampler/MetropolisResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/MetropolisResamplerGPU.cuh"
#endif

template<class V1, class V2, class O1>
void bi::MetropolisResampler::resample(Random& rng, V1 lws, V2 as, O1& s) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  ancestorsPermute(rng, lws, as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2>
void bi::MetropolisResampler::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,MetropolisResamplerGPU,
      MetropolisResamplerHost>::type impl;
  impl::ancestors(rng, lws, as, B);
}

template<class V1, class V2>
void bi::MetropolisResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as) throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,MetropolisResamplerGPU,
      MetropolisResamplerHost>::type impl;
  impl::ancestorsPermute(rng, lws, as, B);
}

template<class V1, class V2>
void bi::MetropolisResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int P) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(P >= 0);
  BI_ASSERT(lws.size() == os.size());

  typename sim_temp_vector<V2>::type as(P);
  ancestors(rng, lws, as);
  ancestorsToOffspring(as, os);
}

#endif
