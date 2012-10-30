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

namespace bi {
/**
 * MultinomialResampler implementation on host.
 */
class MultinomialResamplerHost {
public:
  /**
   * Select ancestors, sorted in ascending order by construction. The basis
   * of this implementation is the generation of sorted random variates
   * using the method of @ref Bentley1979 "Bentley & Saxe (1979)".
   */
  template<class V1, class V2, class V3, class V4>
  static void ancestors(Random& rng, const V1 lws, V2 as, int P, bool sort, bool sorted,
      V3 slws, V4 ps, V3 Ws) throw (ParticleFilterDegeneratedException);
};

/**
 * MultinomialResampler implementation on device.
 */
class MultinomialResamplerGPU {
public:
  /**
   * @copydoc MultinomialResampler::ancestors()
   */
  template<class V1, class V2, class V3, class V4>
  static void ancestors(Random& rng, const V1 lws, V2 as, int P, bool sort, bool sorted,
      V3 slws, V4 ps, V3 Ws) throw (ParticleFilterDegeneratedException);
};

/**
 * Multinomial resampler for particle filter.
 *
 * @ingroup method_resampler
 */
class MultinomialResampler : public Resampler {
public:
  /**
   * Constructor.
   *
   * @param sort True to pre-sort weights, false otherwise.
   */
  MultinomialResampler(const bool sort = true);

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
  void resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as,
      O1& s) throw (ParticleFilterDegeneratedException);
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

  template<class V1, class V2, class V3, class V4>
  void ancestors(Random& rng, const V1 lws, V2 as,
      int P, int ka, int k, bool sorted, V3 lws1, V4 ps, V3 Ws)
      throw (ParticleFilterDegeneratedException);

  template<class V1, class V2, class V3, class V4>
  void ancestors(Random& rng, const V1 lws, V2 as, int P, bool sorted,
      V3 slws, V4 ps, V3 Ws)
      throw (ParticleFilterDegeneratedException);

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

}

#include "../host/resampler/MultinomialResamplerHost.hpp"
#ifdef ENABLE_GPU
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
  copy(as, s);
}

template<class V1, class V2, class V3, class O1>
void bi::MultinomialResampler::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, O1& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  const int P = qlws.size();
  BI_ASSERT(qlws.size() == P);
  BI_ASSERT(lws.size() == P);
  BI_ASSERT(as.size() == P);
  BI_ASSERT(a >= 0 && a < P);

  ancestors(rng, qlws, as);
  set_elements(subrange(as, 0, 1), a);
  permute(as);
  correct(as, qlws, lws);
  copy(as, s);
}

template<class V1, class V2>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(lws.size() == as.size());

  const int P = lws.size();
  typename sim_temp_vector<V1>::type lws1(P), Ws(P);
  typename sim_temp_vector<V2>::type ps(P);

  ancestors(rng, lws, as, P, false, lws1, ps, Ws);
}

template<class V1, class V2, class V3, class V4>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as,
    int P, int ka, int k, bool sorted, V3 lws1, V4 ps, V3 Ws)
    throw (ParticleFilterDegeneratedException) {
  ancestors(rng, lws, as, P, sorted, lws1, ps, Ws);
  set_elements(subrange(as, k, 1), ka);
}

template<class V1, class V2, class V3, class V4>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as,
    int P, bool sorted, V3 lws1, V4 ps, V3 Ws)
    throw (ParticleFilterDegeneratedException) {
  /* pre-conditions */
  assert (V1::on_device == V2::on_device);
  assert (V1::on_device == V3::on_device);
  assert (V1::on_device == V4::on_device);

  typedef typename boost::mpl::if_c<V1::on_device,
      MultinomialResamplerGPU,
      MultinomialResamplerHost>::type impl;
  impl::ancestors(rng, lws, as, P, sort, sorted, lws1, ps, Ws);
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
