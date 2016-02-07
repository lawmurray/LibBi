/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_MULTINOMIALRESAMPLER_HPP
#define BI_RESAMPLER_MULTINOMIALRESAMPLER_HPP

#include "ScanResampler.hpp"
#include "../random/Random.hpp"
#include "../cuda/cuda.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Multinomial resampler for particle filter.
 *
 * @ingroup method_resampler
 */
class MultinomialResampler: public ScanResampler {
public:
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
      ScanResamplerPrecompute<L>& pre)
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
      ScanResamplerPrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * Select offspring.
   *
   * @tparam V1 Vector type.
   * @tparam V2 Integer vector type.
   *
   * @param[in,out] rng Random number generator.
   * @param lws Log-weights.
   * @param P Number of offspring.
   * @param[out] os Offspring.
   */
  template<class V1, class V2, Location L>
  void offspring(Random& rng, const V1 lws, const int P, V2 os,
      ScanResamplerPrecompute<L>& pre) throw (ParticleFilterDegeneratedException);

};

/**
 * @internal
 */
template<Location L>
struct precompute_type<MultinomialResampler,L> {
  typedef ScanResamplerPrecompute<L> type;
};

}

#include "../host/resampler/MultinomialResamplerHost.hpp"
#ifdef __CUDACC__
#include "../cuda/resampler/MultinomialResamplerGPU.hpp"
#endif

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as,
    ScanResamplerPrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
#ifdef __CUDACC__
  typedef typename boost::mpl::if_c<L,MultinomialResamplerGPU,
  MultinomialResamplerHost>::type impl;
#else
  typedef MultinomialResamplerHost impl;
#endif
  impl::ancestors(rng, lws, as, pre);
}

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, ScanResamplerPrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  ancestors(rng, lws, as, pre);
  permute(as);
}

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::offspring(Random& rng, const V1 lws, const int P, V2 os,
    ScanResamplerPrecompute<L>& pre) throw (ParticleFilterDegeneratedException) {
  typename sim_temp_vector<V1>::type as(P);
  ancestors(rng, lws, as, pre);
  ancestorsToOffspring(as, os);
}

#endif
