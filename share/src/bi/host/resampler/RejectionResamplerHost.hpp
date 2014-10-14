/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_REJECTIONRESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_REJECTIONRESAMPLERHOST_HPP

#include "ResamplerHost.hpp"

namespace bi {
/**
 * RejectionResampler implementation on host.
 */
class RejectionResamplerHost: public ResamplerHost {
public:
  /**
   * @copydoc RejectionResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);

  /**
   * @copydoc RejectionResampler::ancestorsPermute()
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as,
      const typename V1::value_type maxLogWeight);
};
}

template<class V1, class V2>
void bi::RejectionResamplerHost::ancestors(Random& rng, const V1 lws,
    V2 as, const typename V1::value_type maxLogWeight) {
  /* pre-condition */
  BI_ASSERT(max_reduce(lws) <= maxLogWeight);

  typedef typename V1::value_type T1;

  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw
  const T1 zero = 0.0;
  const T1 maxWeight = bi::exp(maxLogWeight);

  #pragma omp parallel
  {
    real alpha, lw2;
    int p, p2;

    #pragma omp for
    for (p = 0; p < P2; ++p) {
      /* first proposal */
      if (p < P2/P1*P1) {
        /* death jump (stratified uniform) proposal */
        p2 = p % P1;
      } else {
        /* random proposal */
        p2 = rng.uniformInt(0, P1 - 1);
      }
      lw2 = lws(p2);
      alpha = bi::log(rng.uniform(zero, maxWeight));

      /* rejection loop */
      while (alpha > lw2) {
        p2 = rng.uniformInt(0, P1 - 1);
        lw2 = lws(p2);
        alpha = bi::log(rng.uniform(zero, maxWeight));
      }

      /* write result */
      as(p) = p2;
    }
  }
}

template<class V1, class V2>
void bi::RejectionResamplerHost::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, const typename V1::value_type maxLogWeight) {
  ancestors(rng, lws, as, maxLogWeight);
  permute(as);
}

#endif
