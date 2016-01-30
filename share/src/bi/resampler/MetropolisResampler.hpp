/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_METROPOLISRESAMPLER_HPP
#define BI_RESAMPLER_METROPOLISRESAMPLER_HPP

#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Metropolis resampler for particle filter.
 *
 * @ingroup method_resampler
 *
 * Implements the Metropolis resampler as described in @ref Murray2011a
 * "Murray (2011)" and @ref Murray2014 "Murray, Lee & Jacob (2014)".
 */
class MetropolisResampler {
public:
  /**
   * Constructor.
   *
   * @param B Number of Metropolis steps to take.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   */
  MetropolisResampler(const int B = 0);

  /**
   * Get number of steps.
   */
  int getSteps() const;

  /**
   * Set number of steps.
   */
  void setSteps(const int B);

  /**
   * @copydoc MultinomialResampler::ancestors
   */
  template<class V1, class V2, Location L>
  void ancestors(Random& rng, const V1 lws, V2 as,
      ResamplerPrecompute<L>& pre) throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc MultinomialResampler::ancestorsPermute
   */
  template<class V1, class V2, Location L>
  void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      ResamplerPrecompute<L>& pre) throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc MultinomialResampler::offspring
   */
  template<class V1, class V2, Location L>
  void offspring(Random& rng, const V1 lws, const int P, V2 os,
      ResamplerPrecompute<L>& pre) throw (ParticleFilterDegeneratedException);

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
  typedef ResamplerPrecompute<L> type;
};
}

#include "../host/resampler/MetropolisResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/MetropolisResamplerGPU.cuh"
#endif
#include "../math/sim_temp_vector.hpp"

inline bi::MetropolisResampler::MetropolisResampler(const int B) :
    B(B) {
  //
}

inline int bi::MetropolisResampler::getSteps() const {
  return B;
}

inline void bi::MetropolisResampler::setSteps(const int B) {
  this->B = B;
}

template<class V1, class V2, bi::Location L>
void bi::MetropolisResampler::ancestors(Random& rng, const V1 lws, V2 as,
    ResamplerPrecompute<L>& pre) throw (ParticleFilterDegeneratedException) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,MetropolisResamplerGPU,
  MetropolisResamplerHost>::type impl;
#else
  typedef MetropolisResamplerHost impl;
#endif
  impl::ancestors(rng, lws, as, B);
}

template<class V1, class V2, bi::Location L>
void bi::MetropolisResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, ResamplerPrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<V1::on_device,MetropolisResamplerGPU,
  MetropolisResamplerHost>::type impl;
#else
  typedef MetropolisResamplerHost impl;
#endif
  impl::ancestorsPermute(rng, lws, as, B);
}

template<class V1, class V2, bi::Location L>
void bi::MetropolisResampler::offspring(Random& rng, const V1 lws, const int P, V2 os,
    ResamplerPrecompute<L>& pre) throw (ParticleFilterDegeneratedException) {
  typename sim_temp_vector<V1>::type as(P);
  ancestors(rng, lws, as);
  ancestorsToOffspring(as, os);
}

#endif
