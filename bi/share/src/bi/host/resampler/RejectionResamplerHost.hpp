/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_RESAMPLER_REJECTIONRESAMPLERHOST_HPP
#define BI_HOST_RESAMPLER_REJECTIONRESAMPLERHOST_HPP

template<class V1, class V2>
void bi::RejectionResamplerHost::ancestors(Random& rng, const V1 lws,
    V2 as) {
  BI_ASSERT_MSG(false, "Not yet implemented");

  const int P1 = lws.size(); // number of particles
  const int P2 = as.size(); // number of ancestors to draw

  #pragma omp parallel
  {
    real alpha, lw1, lw2;
    int p1, p2, p;

    #pragma omp for
    for (p = 0; p < P2; ++p) {

      /* write result */
      as(p) = p1;
    }
  }
}

#endif
