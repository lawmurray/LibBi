/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_MULTINOMIALRESAMPLERGPU_HPP
#define BI_CUDA_RESAMPLER_MULTINOMIALRESAMPLERGPU_HPP

template<class V1, class V2>
void bi::MultinomialResamplerGPU::ancestors(Random& rng, const V1 lws, V2 as,
    MultinomialPrecompute<ON_DEVICE>& pre)
        throw (ParticleFilterDegeneratedException) {
  typedef typename V1::value_type T1;

  const int P = as.size();
  const int lwsSize = lws.size();

  typename sim_temp_vector<V1>::type alphas(P);
  typename sim_temp_vector<V2>::type as1(P);

  if (pre.W > 0) {
    /* random numbers */
    rng.uniforms(alphas, 0.0, pre.W);

    /* sample */
    if (pre.sort) {
      bi::lower_bound(pre.Ws, alphas, as1);
      bi::gather(as1, pre.ps, as);
    } else {
      bi::lower_bound(pre.Ws, alphas, as);
    }
  } else {
    throw ParticleFilterDegeneratedException();
  }

  /* post-condition */
  //BI_ASSERT(max_reduce(as) < lws.size());
  // ^ CUDA craziness, doesn't like the above at runtime
  #ifndef NDEBUG
  bool cond = max_reduce(as) < lws.size();
  BI_ASSERT(cond);
  #endif
}

#endif
