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
 * MultinomialResampler implementation on host.
 */
class MultinomialResamplerHost: public ResamplerHost {
public:
  /**
   * Select ancestors, sorted in ascending order by construction. The basis
   * of this implementation is the generation of sorted random variates
   * using the method of @ref Bentley1979 "Bentley & Saxe (1979)".
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as,
      MultinomialPrecompute<ON_HOST>& pre)
          throw (ParticleFilterDegeneratedException);
};

/**
 * MultinomialResampler implementation on device.
 */
class MultinomialResamplerGPU: public ResamplerGPU {
public:
  /**
   * @copydoc MultinomialResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as,
      MultinomialPrecompute<ON_DEVICE>& pre)
          throw (ParticleFilterDegeneratedException);
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
   */
  MultinomialResampler(const bool sort = true, const double essRel = 0.5);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(Random&, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  void resample(Random& rng, V1 lws, V2 as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const V1, V2, V3, O1&)
   */
  template<class V1, class V2, class V3, class O1>
  void resample(Random& rng, const V1 qlws, V2 lws, V3 as, O1& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, V1, V2, O1&)
   */
  template<class V1, class V2, class O1>
  void cond_resample(Random& rng, const int ka, const int k, V1 lws, V2 as,
      O1& s) throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, const V1, V2, V3, O1&)
   */
  template<class V1, class V2, class V3, class O1>
  void resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as, O1& s)
      throw (ParticleFilterDegeneratedException);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::ancestors
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, Location L>
  void ancestors(Random& rng, const V1 lws, V2 as,
      MultinomialPrecompute<L>& pre)
          throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, Location L>
  void precompute(const V1 lws, const V2 as, MultinomialPrecompute<L>& pre);

  /**
   * @copydoc concept::Resampler::offspring
   */
  template<class V1, class V2>
  void offspring(Random& rng, const V1 lws, V2 os, const int P)
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

#include "thrust/sequence.h"
#include "thrust/fill.h"
#include "thrust/binary_search.h"
#include "thrust/scan.h"
#include "thrust/gather.h"

template<class V1, class V2, class O1>
void bi::MultinomialResampler::resample(Random& rng, V1 lws, V2 as, O1& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  ancestors(rng, lws, as);
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class O1>
void bi::MultinomialResampler::cond_resample(Random& rng, const int ka,
    const int k, V1 lws, V2 as, O1& s)
        throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());
  BI_ASSERT(k >= 0 && k < as.size());
  BI_ASSERT(ka >= 0 && ka < lws.size());

  int P;
  if (!sort) {
    // change this?
    P = 0;
  } else {
    P = s.size();
  }
  typename sim_temp_vector<V1>::type lws1(P), Ws(P);
  typename sim_temp_vector<V2>::type ps(P);

  ancestors(rng, lws, as, lws.size(), ka, k, false, lws1, ps, Ws);
  BI_ASSERT(*(as.begin() + k) == ka);
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class V3, class O1>
void bi::MultinomialResampler::resample(Random& rng, const V1 qlws, V2 lws,
    V3 as, O1& s) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  const int P = qlws.size();
  BI_ASSERT(qlws.size() == P);
  BI_ASSERT(lws.size() == P);
  BI_ASSERT(as.size() == P);

  ancestors(rng, qlws, as);
  permute(as);
  correct(as, qlws, lws);
  normalise(lws);
  copy(as, s);
}

template<class V1, class V2, class V3, class O1>
void bi::MultinomialResampler::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, O1& s)
        throw (ParticleFilterDegeneratedException) {
  /* pre-conditions */
  const int P = qlws.size();
  BI_ASSERT(qlws.size() == P);
  BI_ASSERT(lws.size() == P);
  BI_ASSERT(as.size() == P);
  BI_ASSERT(a >= 0 && a < P);

  ancestors(rng, qlws, as);
  set_elements(subrange(as, 0, 1), a);
  permute(as);
  correct(as, qlws, lws);
  normalise(lws);
  copy(as, s);
}

template<class V1, class V2>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  MultinomialPrecompute<V1::location> pre;
  precompute(lws, as, pre);
  ancestors(rng, lws, as, pre);
}

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as,
    MultinomialPrecompute<L>& pre) throw (ParticleFilterDegeneratedException) {
  typedef typename boost::mpl::if_c<L,MultinomialResamplerGPU,
      MultinomialResamplerHost>::type impl;
  impl::ancestors(rng, lws, as, pre);
}

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::precompute(const V1 lws, const V2 as,
    MultinomialPrecompute<L>& pre) {
  const int P = lws.size();
  real lZ;

  pre.Ws.resize(P, false);
  if (sort) {
    pre.lws1.resize(P, false);
    pre.ps.resize(P, false);

    pre.lws1 = lws;
    seq_elements(pre.ps, 0);
    bi::sort_by_key(pre.lws1, pre.ps);
    lZ = sumexpu_inclusive_scan(pre.lws1, pre.Ws);
    pre.W = *(pre.Ws.end() - 1);  // log sum of weights
  } else {
    lZ = sumexpu_inclusive_scan(lws, pre.Ws);
    pre.W = *(pre.Ws.end() - 1);  // log sum of weights
  }
  pre.sort = sort;
}

template<class V1, class V2>
void bi::MultinomialResampler::offspring(Random& rng, const V1 lws, V2 os,
    const int P) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(P >= 0);
  BI_ASSERT(lws.size() == os.size());

  typename sim_temp_vector<V2>::type as(P);
  ancestors(rng, lws, as);
  ancestorsToOffspring(as, os);
}

#endif
