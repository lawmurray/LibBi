/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_REJECTIONRESAMPLER_HPP
#define BI_RESAMPLER_REJECTIONRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * RejectionResampler implementation on host.
 */
class RejectionResamplerHost: public ResamplerBaseHost {
public:
  /**
   * @copydoc RejectionResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);

  /**
   * @copydoc RejectionResampler::ancestorsPermute()
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);
};

/**
 * RejectionResampler implementation on device.
 */
class RejectionResamplerGPU: public ResamplerBaseGPU {
public:
  /**
   * @copydoc MultinomialResampler::ancestors
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);

  /**
   * @copydoc MultinomialResampler::ancestorsPermute
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);
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
class RejectionResampler: public ResamplerBase {
public:
  /**
   * Constructor.
   */
  RejectionResampler();

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
   * @param maxLogWeight Maximum log-weight.
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc ancestors()
   */
  template<class V1, class V2>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight)
          throw (ParticleFilterDegeneratedException);
  //@}
};

/**
 * @internal
 */
template<Location L>
struct precompute_type<RejectionResampler,L> {
  typedef ResamplerBasePrecompute<L> type;
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

template<class V1, class V2>
void bi::RejectionResampler::ancestors(Random& rng, const V1 lws, V2 as,
    const typename V1::value_type maxLogWeight)
        throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,RejectionResamplerGPU,
      RejectionResamplerHost>::type impl;
  impl::ancestors(rng, lws, as, maxLogWeight);
}

template<class V1, class V2>
void bi::RejectionResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, const typename V1::value_type maxLogWeight)
        throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<V1::on_device,RejectionResamplerGPU,
      RejectionResamplerHost>::type impl;
  impl::ancestorsPermute(rng, lws, as, maxLogWeight);
}

#endif
