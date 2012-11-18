/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_MULTINOMIALRESAMPLERGPU_HPP
#define BI_CUDA_RESAMPLER_MULTINOMIALRESAMPLERGPU_HPP

template<class V1, class V2, class V3, class V4, class V5>
void bi::MultinomialResamplerGPU::ancestors(Random& rng, const V1 lws, V2 as,
    int P, bool sort, bool sorted, V3 lws1, V4 ps, V5 Ws)
    throw (ParticleFilterDegeneratedException) {
  /* pre-condition */
  BI_ASSERT(as.size() == P);

  typedef typename V1::value_type T1;
  const int lwsSize = lws.size();

  typename sim_temp_vector<V1>::type alphas(P);
  typename sim_temp_vector<V2>::type as1(P);
  T1 W;

  /* weights */
  if (sort) {
    if (!sorted) {
      lws1 = lws;
      seq_elements(ps, 0);
      bi::sort_by_key(lws1, ps);
      sumexpu_inclusive_scan(lws1, Ws);
    }
  } else {
    sumexpu_inclusive_scan(lws, Ws);
  }
  W = *(Ws.end() - 1);  // sum of weights
  if (W > 0) {
    /* random numbers */
    rng.uniforms(alphas, 0.0, W);

    /* sample */
    if (sort) {
      bi::lower_bound(Ws, alphas, as1);
      bi::gather(as1, ps, as);
    } else {
      bi::lower_bound(Ws, alphas, as);
    }
  } else {
    throw ParticleFilterDegeneratedException();
  }

  /* post-condition */
  BI_ASSERT(max_reduce(as) < lws.size());
}

#endif
