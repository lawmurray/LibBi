/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_RESAMPLER_MULTINOMIALRESAMPLERGPU_HPP
#define BI_CUDA_RESAMPLER_MULTINOMIALRESAMPLERGPU_HPP

#include "ResamplerGPU.cuh"

namespace bi {
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
      ScanResamplerPrecompute<ON_DEVICE>& pre)
          throw (ParticleFilterDegeneratedException);
};
}

template<class V1, class V2>
void bi::MultinomialResamplerGPU::ancestors(Random& rng, const V1 lws, V2 as,
    ScanResamplerPrecompute<ON_DEVICE>& pre)
        throw (ParticleFilterDegeneratedException) {
  typedef typename V1::value_type T1;

  const int P = as.size();
  const int lwsSize = lws.size();

  typename sim_temp_vector<V1>::type alphas(P);
  typename sim_temp_vector<V2>::type as1(P);

  if (pre.W > 0) {
    rng.uniforms(alphas, 0.0, pre.W);
    bi::lower_bound(pre.Ws, alphas, as);
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
