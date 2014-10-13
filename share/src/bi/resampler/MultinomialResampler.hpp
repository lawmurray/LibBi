/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_RESAMPLER_MULTINOMIALRESAMPLER_HPP
#define BI_RESAMPLER_MULTINOMIALRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"
#include "../math/loc_temp_vector.hpp"

namespace bi {
/**
 * Precomputations for MultinomialResampler.
 */
template<Location L>
struct MultinomialPrecompute {
  typename loc_temp_vector<L,real>::type lws1, Ws;
  typename loc_temp_vector<L,int>::type ps;
  real W;
  bool sort;
};

/**
 * Multinomial resampler for particle filter.
 *
 * @ingroup method_resampler
 */
class MultinomialResampler: public Resampler {
public:
  /**
   * Constructor.
   *
   * @param sort True to pre-sort weights, false otherwise.
   * @param essRel Minimum ESS, as proportion of total number of particles,
   * to trigger resampling.
   * @param bridgeEssRel Minimum ESS, as proportion of total number of
   * particles, to trigger resampling after bridge weighting.
   */
  MultinomialResampler(const bool sort = true, const double essRel = 0.5,
      const double bridgeEssRel = 0.5);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc Resampler::resample
   */
  template<class S1>
  void resample(Random& rng, const ScheduleElement now, S1& s)
      throw (ParticleFilterDegeneratedException);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc Resampler::precompute
   */
  template<class V1, Location L>
  void precompute(const V1 lws, MultinomialPrecompute<L>& pre);

  /**
   * @copydoc Resampler::ancestors
   */
  template<class V1, class V2, Location L>
  void ancestors(Random& rng, const V1 lws, V2 as,
      MultinomialPrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc Resampler::offspring
   */
  template<class V1, class V2, Location L>
  void offspring(Random& rng, const V1 lws, V2 os, const int P,
      MultinomialPrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);
  //@}

protected:
  /**
   * Pre-sort weights?
   */
  bool sort;
};

/**
 * @internal
 */
template<Location L>
struct precompute_type<MultinomialResampler,L> {
  typedef MultinomialPrecompute<L> type;
};

}

#include "../host/resampler/MultinomialResamplerHost.hpp"
#ifdef ENABLE_CUDA
#include "../cuda/resampler/MultinomialResamplerGPU.hpp"
#endif

#include "../math/temp_vector.hpp"
#include "../math/sim_temp_vector.hpp"

template<class S1>
void bi::MultinomialResampler::resample(Random& rng,
    const ScheduleElement now, S1& s)
        throw (ParticleFilterDegeneratedException) {
  double lW;
  if (isTriggered(now, s, &lW)) {
    MultinomialPrecompute<V1::location> pre;
    precompute(s.logWeights(), pre);
    ancestors(rng, s.logWeights(), s.ancestors(), pre);
    permute(s.ancestors());
    copy(s.ancestors(), s.getDyn());
    s.logWeights().clear();
    s.logLikelihood += lW;
  }
}

template<class V1, bi::Location L>
void bi::MultinomialResampler::precompute(const V1 lws,
    MultinomialPrecompute<L>& pre) {
  const int P = lws.size();
  typename V1::value_type lZ;

  pre.Ws.resize(P, false);
  if (sort) {
    pre.lws1.resize(P, false);
    pre.ps.resize(P, false);
    pre.lws1 = lws;
    seq_elements(pre.ps, 0);
    bi::sort_by_key(pre.lws1, pre.ps);
    lZ = sumexpu_inclusive_scan(pre.lws1, pre.Ws);
    pre.W = *(pre.Ws.end() - 1);  // sum of weights
  } else {
    lZ = sumexpu_inclusive_scan(lws, pre.Ws);
    pre.W = *(pre.Ws.end() - 1);  // sum of weights
  }
  pre.sort = sort;
}

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as,
    MultinomialPrecompute<L>& pre) throw (ParticleFilterDegeneratedException) {
#ifdef ENABLE_CUDA
  typedef typename boost::mpl::if_c<L,MultinomialResamplerGPU,
  MultinomialResamplerHost>::type impl;
#else
  typedef MultinomialResamplerHost impl;
#endif
  impl::ancestors(rng, lws, as, pre);
}

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int P, MultinomialPrecompute<L>& pre)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(P >= 0);
  BI_ASSERT(lws.size() == os.size());

  typename sim_temp_vector<V2>::type as(P);
  ancestors(rng, lws, as, pre);
  ancestorsToOffspring(as, os);
}

#endif
