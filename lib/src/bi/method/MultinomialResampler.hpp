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
   * @param rng Random number generator.
   * @param sort True to pre-sort weights, false otherwise.
   */
  MultinomialResampler(Random& rng, const bool sort = true);

  /**
   * @name High-level interface
   */
  //@{
  /**
   * @copydoc concept::Resampler::resample(V1&, V2&)
   */
  template<class V1, class V2, Location L>
  void resample(V1& lws, V2& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3, Location L>
  void resample(const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, V1&, V2&)
   */
  template<class V1, class V2, Location L>
  void resample(const int a, V1& lws, V2& as, Static<L>& theta, State<L>& s);

  /**
   * @copydoc concept::Resampler::resample(const typename V2::value_type, const V1&, V1&, V2&)
   */
  template<class V1, class V2, class V3, Location L>
  void resample(const int a, const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s);
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
   * @param lws Log-weights.
   * @param[out] as Ancestry.
   */
  template<class V1, class V2>
  void ancestors(const V1& lws, V2& as);
  //@}

private:
  /**
   * %Random number generator.
   */
  Random& rng;

  /**
   * Pre-sort weights?
   */
  bool sort;
};

}

#include "../cuda/math/temp_vector.hpp"

#include "thrust/sequence.h"
#include "thrust/fill.h"
#include "thrust/binary_search.h"
#include "thrust/scan.h"
#include "thrust/gather.h"

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::resample(V1& lws, V2& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  assert (lws.size() == as.size());

  ancestors(lws, as);
  permute(as);
  copy(as, theta, s);
  lws.clear();
}

template<class V1, class V2, bi::Location L>
void bi::MultinomialResampler::resample(const int a, V1& lws, V2& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  ancestors(lws, as);
  as[0] = a;
  permute(as);
  copy(as, theta, s);
  lws.clear();
}

template<class V1, class V2, class V3, bi::Location L>
void bi::MultinomialResampler::resample(const V1& qlws, V2& lws, V3& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);

  ancestors(qlws, as);
  permute(as);
  correct(as, qlws, lws);
  copy(as, theta, s);
}

template<class V1, class V2, class V3, bi::Location L>
void bi::MultinomialResampler::resample(const int a, const V1& qlws,
    V2& lws, V3& as, Static<L>& theta, State<L>& s) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);
  assert (a >= 0 && a < P);

  ancestors(qlws, as);
  as[0] = a;
  permute(as);
  correct(as, qlws, lws);
  copy(as, theta, s);
}

template<class V1, class V2>
void bi::MultinomialResampler::ancestors(const V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());

  typedef typename V1::value_type T1;
  const int P = lws.size();

  BOOST_AUTO(lws1, temp_vector<V1>(P));
  BOOST_AUTO(as1, temp_vector<V2>(P));
  BOOST_AUTO(Ws, temp_vector<V1>(P));
  BOOST_AUTO(ps, temp_vector<V2>(P));
  BOOST_AUTO(hostAlphas, host_temp_vector<T1>(P));
  T1 W;

  /* weights */
  if (sort) {
    *lws1 = lws;
    bi::sequence(ps->begin(), ps->end(), 0);
    bi::sort_by_key(lws1->begin(), lws1->end(), ps->begin());
    bi::inclusive_scan_sum_exp(lws1->begin(), lws1->end(), Ws->begin());
  } else {
    bi::inclusive_scan_sum_exp(lws.begin(), lws.end(), Ws->begin());
  }
  W = *(Ws->end() - 1); // sum of weights
  if (W > 0) {
    /* random numbers */
    rng.uniforms(hostAlphas, 0.0, W);
    BOOST_AUTO(alphas, map_vector(lws, *hostAlphas));

    /* sample */
    if (sort) {
      thrust::upper_bound(Ws->begin(), Ws->end(), alphas->begin(), alphas->end(), as1->begin());
      thrust::gather(as1->begin(), as1->end(), ps->begin(), as.begin());
    } else {
      thrust::upper_bound(Ws->begin(), Ws->end(), alphas->begin(), alphas->end(), as.begin());
    }

    delete alphas;
  } else {
    BI_ERROR(W > 0, "Particle filter has degenerated");
  }

  synchronize();
  delete lws1;
  delete as1;
  delete Ws;
  delete ps;
  delete hostAlphas;
}

#endif
