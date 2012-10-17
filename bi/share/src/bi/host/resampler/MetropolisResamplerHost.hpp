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
  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw

  #pragma omp parallel
  {
    real alpha, lw1, lw2;
    int c, p1, p2, p;

    #pragma omp for
    for (p = 0; p < P2; ++p) {
      p1 = p;
      lw1 = lws(p);
      for (c = 0; c < C; ++c) {
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

#endif
