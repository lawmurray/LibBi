/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_METROPOLISRESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_METROPOLISRESAMPLERHOST_HPP

template<class V1, class V2>
void bi::MetropolisResamplerHost::ancestors(Random& rng, const V1 lws,
    V2 as, int C) {
  const int P = lws.size();

  #pragma omp parallel
  {
    real alpha, lw1, lw2;
    int c, p1, p2, tid;

    #pragma omp for
    for (tid = 0; tid < P; ++tid) {
      p1 = tid;
      lw1 = lws[tid];
      for (c = 0; c < C; ++c) {
        p2 = rng.uniformInt(0, P - 1);
        lw2 = lws[p2];
        alpha = rng.uniform<real>();

        if (alpha < bi::exp(lw2 - lw1)) {
          /* accept */
          p1 = p2;
          lw1 = lw2;
        }
      }

      /* write result */
      as[tid] = p1;
    }
  }
}

#endif
