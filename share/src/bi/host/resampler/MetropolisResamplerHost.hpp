/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_METROPOLISRESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_METROPOLISRESAMPLERHOST_HPP

#include "ResamplerHost.hpp"

namespace bi {
/**
 * MetropolisResampler implementation on host.
 */
class MetropolisResamplerHost: public ResamplerHost {
public:
  /**
   * @copydoc MetropolisResampler::ancestors()
   */
  template<class V1, class V2>
  static void ancestors(Random& rng, const V1 lws, V2 as, int B);

  /**
   * @copydoc MetropolisResampler::ancestorsPermute()
   */
  template<class V1, class V2>
  static void ancestorsPermute(Random& rng, const V1 lws, V2 as, int B);
};
}

template<class V1, class V2>
void bi::MetropolisResamplerHost::ancestors(Random& rng, const V1 lws,
    V2 as, int B) {
  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw

  #pragma omp parallel
  {
    real alpha, lw1, lw2;
    int k, p1, p2, p;

    #pragma omp for
    for (p = 0; p < P2; ++p) {
      p1 = p;
      lw1 = lws(p);
      for (k = 0; k < B; ++k) {
        p2 = rng.uniformInt(0, P1 - 1);
        lw2 = lws(p2);
        alpha = rng.uniform<real>();

        if (bi::log(alpha) < lw2 - lw1) {
          /* accept */
          p1 = p2;
          lw1 = lw2;
        }
      }

      /* write result */
      as(p) = p1;
    }
  }
}

template<class V1, class V2>
void bi::MetropolisResamplerHost::ancestorsPermute(Random& rng, const V1 lws,
    V2 as, int B) {
  ancestors(rng, lws, as, B);
  permute(as);
}

#endif
