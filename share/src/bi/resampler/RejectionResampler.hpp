/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_REJECTIONRESAMPLER_HPP
#define BI_RESAMPLER_REJECTIONRESAMPLER_HPP

#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Precomputed results for RejectionResampler.
 */
struct RejectionResamplerPrecompute {
  double maxLogWeight;
};

/**
 * Rejection resampler for particle filter.
 *
 * @ingroup method_resampler
 *
 * Implements the rejection resampler as described in
 * @ref Murray2014 "Murray, Lee & Jacob (2014)".
 *
 * Unlike other resamplers, the rejection resampler does not use an
 * ESS threshold to determine whether or not to resample at each time.
 * Instead, it simply samples at every time step. This is for two reasons:
 *
 * @li it is only possible in the current implementation to compute a (good) bound
 * on the incremental log-likelihood for a single time point, not accumulated
 * across multiple time points when resampling may be skipped at certain
 * times, and
 *
 * @li computing ESS is a collective operation, which defeats the main
 * motivation of the rejection resampler (and, indeed, the Metropolis
 * resampler), which is precisely to avoid these.
 */
class RejectionResampler {
public:
  /**
   * @copydoc Multinomial::ancestors()
   */
  template<class V1, class V2, bi::Location L>
  void ancestors(Random& rng, const V1 lws, V2 as,
      RejectionResamplerPrecompute& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc Multinomial::ancestorsPermute()
   */
  template<class V1, class V2, bi::Location L>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      RejectionResamplerPrecompute& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc MultinomialResampler::offspring
   */
  template<class V1, class V2, Location L>
  void offspring(Random& rng, const V1 lws, const int P, V2 os,
      RejectionResamplerPrecompute& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc Resampler::precompute
   */
  template<class V1>
  void precompute(const V1 lws, RejectionResamplerPrecompute& pre);
};

/**
 * @internal
 */
template<Location L>
struct precompute_type<RejectionResampler,L> {
  typedef RejectionResamplerPrecompute type;
};

/**
 * @internal
 */
template<>
struct resampler_needs_max<RejectionResampler> {
  static const bool value = true;
};

}

#include "../host/resampler/RejectionResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/RejectionResamplerGPU.cuh"
#endif

template<class V1, class V2, bi::Location L>
void bi::RejectionResampler::ancestors(Random& rng, const V1 lws, V2 as,
    RejectionResamplerPrecompute& pre)
        throw (ParticleFilterDegeneratedException) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,RejectionResamplerGPU,
  RejectionResamplerHost>::type impl;
#else
  typedef RejectionResamplerHost impl;
#endif
  impl::ancestors(rng, lws, as, pre.maxLogWeight);
}

template<class V1, class V2, bi::Location L>
void bi::RejectionResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, RejectionResamplerPrecompute& pre)
        throw (ParticleFilterDegeneratedException) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,RejectionResamplerGPU,
  RejectionResamplerHost>::type impl;
#else
  typedef RejectionResamplerHost impl;
#endif
  impl::ancestorsPermute(rng, lws, as, pre.maxLogWeight);
}

template<class V1>
void bi::RejectionResampler::precompute(const V1 lws,
    RejectionResamplerPrecompute& pre) {
  BI_ERROR_MSG(false, "Not yet implemented");
}

template<class V1, class V2, bi::Location L>
void bi::RejectionResampler::offspring(Random& rng, const V1 lws, const int P, V2 os,
    RejectionResamplerPrecompute& pre) throw (ParticleFilterDegeneratedException) {
  typename sim_temp_vector<V1>::type as(P);
  ancestors(rng, lws, as);
  ancestorsToOffspring(as, os);
}

#endif
