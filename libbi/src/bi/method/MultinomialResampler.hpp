/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MULTINOMIALRESAMPLER_HPP
#define BI_METHOD_MULTINOMIALRESAMPLER_HPP

#include "Resampler.hpp"
#include "../cuda/cuda.hpp"
#include "../random/Random.hpp"
#include "../misc/exception.hpp"

namespace bi {
/**
 * Multinomial resampler for particle filter.
 *
 * @ingroup method
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
   * @copydoc concept::Resampler::resample(Random&, V1, V2, State<B,L>&)
   */
  template<class V1, class V2, class B, Location L>
  void resample(Random& rng, V1 lws, V2 as, State<B,L>& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const V1, V2, V3, State<B,L>&)
   */
  template<class V1, class V2, class V3, class B, Location L>
  void resample(Random& rng, const V1 qlws, V2 lws, V3 as,
      State<B,L>& s) throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, V1, V2, State<B,L>&)
   */
  template<class V1, class V2, class B, Location L>
  void resample(Random& rng, const int a, V1 lws, V2 as, State<B,L>& s)
      throw (ParticleFilterDegeneratedException);

  /**
   * @copydoc concept::Resampler::resample(Random&, const int, const V1, V2, V3, State<B,L>&)
   */
  template<class V1, class V2, class V3, class B, Location L>
  void resample(Random& rng, const int a, const V1 qlws, V2 lws, V3 as,
      State<B,L>& s)
      throw (ParticleFilterDegeneratedException);
  //@}

  /**
   * @name Low-level interface
   */
  //@{
  /**
   * Select ancestors.
   *
   * @tparam V1 Floating point vector type.
   * @tparam V2 Integer vector type.
   *
   * @param rng Random number generator.
   * @param lws Log-weights.
   * @param[out] as Ancestry.
   */
  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as)
      throw (ParticleFilterDegeneratedException);
  //@}

  template<class V1, class V2>
  void ancestors(Random& rng, const V1 lws, V2 as, int P)
      throw (ParticleFilterDegeneratedException);

private:
  /**
   * Pre-sort weights?
   */
  bool sort;
};

}

#include "../math/temp_vector.hpp"
#include "../math/sim_temp_vector.hpp"

#include "thrust/sequence.h"
#include "thrust/fill.h"
#include "thrust/binary_search.h"
#include "thrust/scan.h"
#include "thrust/gather.h"

template<class V1, class V2, class B, bi::Location L>
void bi::MultinomialResampler::resample(Random& rng, V1 lws, V2 as,
    State<B,L>& s) throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (lws.size() == as.size());

  ancestors(rng, lws, as);
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class B, bi::Location L>
void bi::MultinomialResampler::resample(Random& rng, const int a, V1 lws,
    V2 as, State<B,L>& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  ancestors(rng, lws, as);
  as[0] = a;
  permute(as);
  copy(as, s);
  lws.clear();
}

template<class V1, class V2, class V3, class B, bi::Location L>
void bi::MultinomialResampler::resample(Random& rng, const V1 qlws, V2 lws,
    V3 as, State<B,L>& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);

  ancestors(rng, qlws, as);
  permute(as);
  correct(as, qlws, lws);
  copy(as, s);
}

template<class V1, class V2, class V3, class B, bi::Location L>
void bi::MultinomialResampler::resample(Random& rng, const int a,
    const V1 qlws, V2 lws, V3 as, State<B,L>& s)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);
  assert (a >= 0 && a < P);

  ancestors(rng, qlws, as);
  as[0] = a;
  permute(as);
  correct(as, qlws, lws);
  copy(as, s);
}

template<class V1, class V2>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (lws.size() == as.size());

  typedef typename V1::value_type T1;
  const int P = lws.size();

  typename sim_temp_vector<V1>::type lws1(P), Ws(P), alphas(P);
  typename sim_temp_vector<V2>::type as1(P), ps(P);
  T1 W;

  /* weights */
  if (sort) {
    lws1 = lws;
    seq_elements(ps, 0);
    bi::sort_by_key(lws1, ps);
    bi::inclusive_scan_sum_exp(lws1, Ws);
  } else {
    bi::inclusive_scan_sum_exp(lws, Ws);
  }
  W = *(Ws.end() - 1); // sum of weights
  if (W > 0) {
    /* random numbers */
    rng.uniforms(alphas, 0.0, W);

    /* sample */
    if (sort) {
      thrust::upper_bound(Ws.begin(), Ws.end(), alphas.begin(), alphas.end(), as1.begin());
      thrust::gather(as1.begin(), as1.end(), ps.begin(), as.begin());
    } else {
      thrust::upper_bound(Ws.begin(), Ws.end(), alphas.begin(), alphas.end(), as.begin());
    }
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

template<class V1, class V2>
void bi::MultinomialResampler::ancestors(Random& rng, const V1 lws, V2 as, int P)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  assert (as.size() == P);

  typedef typename V1::value_type T1;
  const int lwsSize = lws.size();

  typename sim_temp_vector<V1>::type lws1(lwsSize), Ws(lwsSize), alphas(P);
  typename sim_temp_vector<V2>::type as1(P), ps(lwsSize);
  T1 W;

  /* weights */
  if (sort) {
    lws1 = lws;
    seq_elements(ps, 0);
    bi::sort_by_key(lws1, ps);
    bi::inclusive_scan_sum_exp(lws1, Ws);
  } else {
    bi::inclusive_scan_sum_exp(lws, Ws);
  }
  W = *(Ws.end() - 1); // sum of weights
  if (W > 0) {
    /* random numbers */
    rng.uniforms(alphas, 0.0, W);

    /* sample */
    if (sort) {
      thrust::upper_bound(Ws.begin(), Ws.end(), alphas.begin(), alphas.end(), as1.begin());
      thrust::gather(as1.begin(), as1.end(), ps.begin(), as.begin());
    } else {
      thrust::upper_bound(Ws.begin(), Ws.end(), alphas.begin(), alphas.end(), as.begin());
    }
  } else {
    throw ParticleFilterDegeneratedException();
  }
}

#endif
