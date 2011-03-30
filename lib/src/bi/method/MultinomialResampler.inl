/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_MULTINOMIALRESAMPLER_INL
#define BI_METHOD_MULTINOMIALRESAMPLER_INL

#include "../cuda/math/temp_vector.hpp"

#include "thrust/sequence.h"
#include "thrust/fill.h"
#include "thrust/binary_search.h"
#include "thrust/scan.h"
#include "thrust/gather.h"

template<class V1, class V2>
void bi::MultinomialResampler::resample(V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());

  ancestors(lws, as);
  permute(as);
  lws.clear();
}

template<class V1, class V2>
void bi::MultinomialResampler::resample(const int a, V1& lws, V2& as) {
  /* pre-condition */
  assert (lws.size() == as.size());
  assert (a >= 0 && a < as.size());

  ancestors(lws, as);
  as[0] = a;
  permute(as);
  lws.clear();
}

template<class V1, class V2, class V3>
void bi::MultinomialResampler::resample(const V1& qlws, V2& lws, V3& as) {
  /* pre-condition */
  const int P = qlws.size();
  assert (qlws.size() == P);
  assert (lws.size() == P);
  assert (as.size() == P);

  ancestors(qlws, as);
  permute(as);
  correct(as, qlws, lws);
}

template<class V1, class V2, class V3>
void bi::MultinomialResampler::resample(const int a, const V1& qlws,
    V2& lws, V3& as) {
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
